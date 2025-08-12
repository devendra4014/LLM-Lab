import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

torch.backends.cudnn.benchmark = True


# create a dataclass which will hold hyperparameters for CharGPT model training
@dataclass
class CharGPTConfig:
    num_epochs: int = 10
    batch_size: int = 8  # number of independent examples
    block_size: int = 128  # maximum tokes handled by llm at a time, we call it as context_length

    learning_rate = 1e-4

    # Model Parameters
    emb_dim: int = 512
    n_heads: int = 8  # number of heads to be calculated in parallel inside attention block
    n_layers: int = 4  # number of attention blocks
    vocab_size: int = -1  # This will be set when we load tokenizer
    dropout: float = 0.2


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, item):
        x = self.data[item: item + self.block_size]
        y = self.data[item + 1: item + self.block_size + 1]
        return x, y


# sinusoidal positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # create positional encoding vector
        pe = torch.zeros(max_len, d_model)

        # positions in columns
        positions = torch.arange(start=0, end=max_len, dtype=torch.float32).unsqueeze(dim=1)

        embedding_index = torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)
        denominator = 1 / torch.tensor(10000.0).pow(embedding_index / d_model)
        # exp_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # print(torch.allclose(exp_term, denominator))

        # pe[:, 0::2]
        # ':' refers to rows from first to last &&
        # '0::2' refers to columns starting from 0th column with step size = 2
        pe[:, 0::2] = torch.sin(positions * denominator)  # even positions
        pe[:, 1::2] = torch.cos(positions * denominator)  # odd positions

        #  "register_buffer()" ensures that 'pe' will be moved to wherever the model gets moved to. So if the
        # model is moved to a GPU, then, even though we don't need to optimize 'pe', it will also be moved to that
        # GPU. # This, in turn, means that accessing 'pe' will be relatively fast compared to having a GPU have to get
        # the data from a CPU.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.size(1) will give number of tokens, so we will get positional encoding only for that number of tokens
        return x + self.pe[:x.size(1), :]


# The normalization is performed as follows:
# 1. Given an input of batch size b, sequence length n, and vector dimension d,
#    calculate the mean and variance across each vector dimension.
# 2. Normalize the input by subtracting the mean and dividing it by the square root of the variance.
#    A small epsilon value is added to the denominator for numerical stability.
# 3. Multiply by a scale parameter (gamma) and add a shift parameter (beta) to the resulting values.
#    These parameters are learned during the training process.
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor):
        # x ==> (B, T, C)
        # (B = Batch Size), (T = `block_size`, or `context_length`), (C = Embedding Dimension 'emb_dim' or 'features')

        x_mean = x.mean(dim=-1, keepdim=True)  # mean across vector dimension ==> x.size(2)
        x_std = x.std(dim=-1, keepdim=True, unbiased=False)  # standard deviation across vector dimension ==> x.size(2)
        # variance = x.var(-1, keepdim=True, unbiased=False)
        # denominator = torch.sqrt(variance + self.eps) <==> (x_std + self.eps)

        x_norm = (x - x_mean) / (x_std + self.eps)
        x = self.gamma * x_norm + self.beta

        return x


class RMSNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.features = feature
        self.gain = nn.Parameter(torch.ones(feature))

    def forward(self, x):
        # Calculate RMS across the last dimension(s)
        # torch.rsqrt() will calculate inverse square root, Hence no need to divide by this term, instead we multiply
        rms = torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize
        x_norm = x * rms * self.gain
        return x_norm


class FeedForward(nn.Module):
    def __init__(self, config:CharGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.emb_dim, config.emb_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()

        self.emb_dim = config.emb_dim  # d_model
        self.dropout = CharGPTConfig.dropout
        self.W_q = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=False)
        self.W_k = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=False)
        self.W_v = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            self.register_buffer("mask", torch.tril(torch.ones(CharGPTConfig.block_size, CharGPTConfig.block_size)))

    def forward(self, x, is_mask=True):
        B, T, C = x.shape  # B, block_size, emb_dim
        q = self.W_q(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)
        k = self.W_k(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)
        v = self.W_v(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)

        if self.flash:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                    dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # (B, block_size, emb_dim) * (B, emb_dim, block_size) -> (B, block_size, block_size)
            attention_score = torch.matmul(q, k.transpose(dim0=-2, dim1=-1))

            # print(f"{k.shape[-1], self.d_model}") # are equal or same
            # (B, block_size, block_size)
            scaled_score = attention_score / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))

            if is_mask:
                # (B, block_size, block_size)
                scaled_score = scaled_score.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

            # print("normalised")
            normalized_score = F.softmax(scaled_score, dim=-1) # (B, block_size, block_size)

            # (B, block_size, block_size) * (B, block_size, emb_dim)  ->  (B, block_size, emb_dim)
            output = torch.matmul(normalized_score, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.head_dim = config.emb_dim // config.n_heads
        self.dropout = config.dropout

        # w_k -> (emb_dim, emb_dim), w_v -> (emb_dim, emb_dim), w_q -> (emb_dim, emb_dim)
        # instead of creating 3 matrix of size (emb_dim, emb_dim)  --> (w_k, w_v, w_q )
        # we can create 1 matrix of size  (emb_dim, 3 * emb_dim)
        self.qkv = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=False)

        # after applying attention we get, (batch, head, block_size, head_dim),
        # and all the heads are concatenated or say stacked one after another,
        # and we get, (B, T, C) <-> (batch, block_size, emb_dim)
        # so all heads are unaware of each other and to tall with each other we multiply
        # finally with 'c_proj' matrix to get => (B, T, C) * (C, C) -> (B, T, C)
        self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # check for flash attention
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # if not self.flash:
        self.register_buffer('mask', torch.tril(torch.ones(size=(1, 1, config.block_size, config.block_size))))

    def forward(self, x):
        # batch_size, token_seq_length (block_size), emb_dim
        B, T, dim = x.shape

        # (B, T, C) *  (C, 3C)  ->  (B, T, 3C)
        qkv: torch.Tensor = self.qkv(x)

        # (B, T, 3C) -> (B, T, C), (B, T, C), (B, T, C)
        q, k, v = qkv.split(self.emb_dim, dim=-1)

        # LET, N = n_heads, D = head_dim
        # Reshape q,k,v into (B, T, N, D) then apply transpose to get (B, N, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(dim0=2, dim1=1)  # (B, N, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(dim0=2, dim1=1)  # (B, N, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(dim0=2, dim1=1)  # (B, N, T, D)

        # K.V ==> (B, N, T, D) * (B, N, D, T)  -> (B, N, T, T)
        attn = q @ k.transpose(-2, -1)

        # scaled dot product ==> divide by square_root of head_dim
        attn = attn * (1/math.sqrt(self.head_dim))  # (B, N, T, T)

        # apply mask over attention
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # (B, N, T, T)

        # apply softmax over last dimension -> head_dim
        attn = F.softmax(attn, dim=-1) # (B, N, T, T)

        # apply dropout only if training
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # multiply ==> (attn * v) ==> (B, N, T, T) * (B, N, T, D)  ->  (B, N, T, D)
        out = attn @ v

        # reshape to (B, T, N, D) and then convert to shape (B, T, C)
        out = out.transpose(2, 1).contiguous().view(B, T, dim)

        # apply last projection layer
        out = self.c_proj(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, config: CharGPTConfig, multi_head: bool = False):
        super().__init__()

        self.ln_1 = LayerNormalization(config.emb_dim)
        self.attention = MultiHeadAttention(config) if multi_head else Attention(config)
        self.ln_2 = LayerNormalization(config.emb_dim)
        self.ff = FeedForward(config)

    def forward(self, x):
        # attention with residual connection
        x = x + self.attention(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))

        return x


# create a CharGPT class which extends torch.nn module and implements transformer architecture
class CharGPT(nn.Module):
    def __init__(self, vocab_size: int, config: CharGPTConfig = CharGPTConfig()):
        super().__init__()
        self.config = config
        self.config.vocab_size = vocab_size

        # embedding layer to convert character ids to embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)  # (block_size, emb_dim)
        self.positional_encoding = PositionalEncoding(config.emb_dim, config.block_size)  # (block_size, emb_dim)
        self.h = nn.ModuleList([AttentionBlock(config, multi_head=True) for _ in range(config.n_layers)])
        self.ln_f = LayerNormalization(config.emb_dim)

        # output layer to convert embeddings back to character ids
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, x, targets=None):
        # `B` is a number of batches getting processed in parallel
        # `T` is the list of tokens in a batch, it is the block_size/context length
        # 'C' is the emb_dim (embedding dimension)

        x = self.token_embedding(x)  # (B,T) ---> (B,T,C) <==> (B, block_size, emb_dim)
        x = self.positional_encoding(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, ids, max_len=1000):
        for i in range(max_len):
            logits, loss = self(ids)  # (batch, token , vocab_size)

            # select last token prediction for a batch of vocab size
            logits = logits[:, -1, :]  # token , vocab_size ==> (1, vocab_size)

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            ids = torch.cat((ids, idx_next), dim=1)

        return ids


def main():
    # read a text file
    with open('../../data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # create a vocabulary which is unique set of characters
    chrs = sorted(list(set(text)))

    # create mapping dictionary for vocabulary which are then used to convert text into ids and ids into text
    char_to_id = {ch: idx for idx, ch in enumerate(chrs)}
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}

    encoder = lambda x: [char_to_id[i] for i in x]
    decoder = lambda y: "".join([id_to_char[i] for i in y])

    # create data of tensor class with long datatype, which is provided to embedding layer first
    text = text[:20000]
    data = torch.tensor(encoder(text), dtype=torch.long)
    # print(data.shape)

    # create train and text split
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    # create train dataset
    train_dataset = CharDataset(train_data, CharGPTConfig.block_size)
    val_dataset = CharDataset(test_data, CharGPTConfig.block_size)

    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=10 * CharGPTConfig.batch_size)
    test_sampler = RandomSampler(train_dataset, replacement=False, num_samples=10 * CharGPTConfig.batch_size)

    train_loader = DataLoader(train_dataset, CharGPTConfig.batch_size)
    val_loader = DataLoader(val_dataset, CharGPTConfig.batch_size)

    device = 'cuda'
    # create a model
    model = CharGPT(len(chrs))
    model.to(device)
    # model = torch.compile(model)

    # create a optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=CharGPTConfig.learning_rate)

    for epoch in range(CharGPTConfig.num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            xb, yb = batch
            logits, loss = model(xb.to(device), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, val_loss = model(xb, yb)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"step {epoch}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

    # save the model
    # torch.save(model.state_dict(), 'models/char_gpt_trained.pth')
    print("training completed")
    start_token = torch.tensor([[char_to_id['H']]]).to(device)
    generated_text = model.generate(start_token)
    print("Generated text:", decoder(generated_text[0].tolist()))


if __name__ == '__main__':
    main()
