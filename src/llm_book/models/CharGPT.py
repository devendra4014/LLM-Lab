import math
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


# create a dataclass which will hold hyperparameters for CharGPT model training
# Added explicit typing for learning_rate to make the config predictable and IDE-friendly.
@dataclass
class CharGPTConfig:
    num_epochs: int = 10
    batch_size: int = 8  # number of independent examples
    block_size: int = 128  # maximum tokens handled by llm at a time (context length)
    learning_rate: float = 1e-4

    # Model Parameters
    emb_dim: int = 512
    n_heads: int = 8  # number of heads to be calculated in parallel inside attention block
    n_layers: int = 4  # number of attention blocks
    vocab_size: int = -1  # This will be set when we instantiate the model
    dropout: float = 0.2


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        # expects a 1-D tensor of token ids
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # number of available contiguous (input, target) pairs
        return len(self.data) - self.block_size

    def __getitem__(self, item):
        # return input sequence x and next-token targets y
        x = self.data[item: item + self.block_size]
        y = self.data[item + 1: item + self.block_size + 1]
        return x, y


# sinusoidal positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # create positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # positions in columns
        positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        indices = torch.arange(0, d_model, step=2, dtype=torch.float32)  # even indexes
        denominator = torch.pow(10000.0, (indices / d_model))  # same as exp(index * -log(10000)/d_model)
        
        # exp_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # print(torch.allclose(exp_term, denominator))

        # pe[:, 0::2]
        # ':' refers to rows from first to last &&
        # '0::2' refers to columns starting from 0th column with step size = 2
        pe[:, 0::2] = torch.sin(positions / denominator)  # even positions
        pe[:, 1::2] = torch.cos(positions / denominator)  # odd positions

        #  "register_buffer()" ensures that 'pe' will be moved to wherever the model gets moved to. So if the
        # model is moved to a GPU, then, even though we don't need to optimize 'pe', it will also be moved to that
        # GPU. 
        # # This, in turn, means that accessing 'pe' will be relatively fast compared to having a GPU have to get
        # the data from a CPU.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, C) -> add positional encodings for first T positions
        # x.size(1) will give number of tokens, so we will get positional encoding only for that number of tokens
        T = x.size(1)
        
        # broadcast (T, C) to (B, T, C)
        return x + self.pe[:T, :]


# LayerNorm implementation that mirrors the description (mean/variance per vector)
# The normalization is performed as follows:
# 1. Given an input of batch size b, sequence length n, and vector dimension d,
#    calculate the mean and variance across each vector dimension.
# 2. Normalize the input by subtracting the mean and dividing it by the square root of the variance.
#    A small epsilon value is added to the denominator for numerical stability.
# 3. Multiply by a scale parameter (gamma) and add a shift parameter (beta) to the resulting values.
#    These parameters are learned during the training process.
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
         # (B = Batch Size), (T = `block_size`, or `context_length`), (C = Embedding Dimension 'emb_dim' or 'features')

        # mean across vector dimension ==> x.size(2)
        mean = x.mean(dim=-1, keepdim=True)

        # standard deviation across vector dimension ==> x.size(2)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        # variance = x.var(-1, keepdim=True, unbiased=False)
        # denominator = torch.sqrt(variance + self.eps) <==> (x_std + self.eps)
        
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta


# RMSNorm (root-mean-square layer norm) as an alternative normalization
class RMSNorm(nn.Module):
    def __init__(self, feature: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.features = feature
        self.gain = nn.Parameter(torch.ones(feature))

    def forward(self, x: torch.Tensor):
        # Compute RMS over last dimension and normalize
        # torch.rsqrt() will calculate inverse square root, Hence no need to divide by this term, instead we multiply
        rms = torch.rsqrt((x.pow(2).mean(dim=-1, keepdim=True)) + self.eps)
        return x * rms * self.gain


class FeedForward(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.emb_dim, config.emb_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()
        self.emb_dim = config.emb_dim  # d_model
        # use the instance config, not the class
        self.dropout = config.dropout

        self.W_q = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_k = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.W_v = nn.Linear(self.emb_dim, self.emb_dim, bias=False)

        # detect if PyTorch provides scaled_dot_product_attention (flash-friendly)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if self.flash:
            # store a causal mask buffer sized by config.block_size to avoid recreating tensors at runtime
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor, is_mask: bool = True):
        # x: (B, T, C) :  (Batch, block_size, emb_dim)
        B, T, C = x.shape
        q = self.W_q(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)
        k = self.W_k(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)
        v = self.W_v(x)  # (B, block_size, emb_dim) *  (emb_dim, emb_dim)  ->  (B, block_size, emb_dim)

        if self.flash:
            # use PyTorch fused attention when available; is_causal enforces autoregressive masking
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                    dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # compute attention scores (B, T, T)
            # (B, block_size, emb_dim) * (B, emb_dim, block_size) -> (B, block_size, block_size)
            scores = torch.matmul(q, k.transpose(-2, -1))
            # scale by sqrt of key dim using Python scalar to avoid device mismatch
            denom = math.sqrt(k.shape[-1])
            scores = scores / denom

            if is_mask:
                # (B, block_size, block_size)
                scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

            weights = F.softmax(scores, dim=-1)  # (B, T, T)
            # (B, block_size, block_size) * (B, block_size, emb_dim)  ->  (B, block_size, emb_dim)
            output = torch.matmul(weights, v)  # (B, T, C)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: CharGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.emb_dim = config.emb_dim
        self.head_dim = config.emb_dim // config.n_heads
        self.dropout = config.dropout

        # single proj for qkv for efficiency
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

        # flash detection (prefer F alias)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # causal mask shaped for broadcasting over (B, N, T, T)
        self.register_buffer('mask', torch.tril(torch.ones(1, 1, config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        
        # (B, T, 3C) -> (B, T, C), (B, T, C), (B, T, C)
        q, k, v = qkv.split(self.emb_dim, dim=-1)

        # reshape to (B, N, T, D)
        # LET, N = n_heads, D = head_dim
        # Reshape q,k,v into (B, T, N, D) then apply transpose to get (B, N, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attn logits (B, N, T, T)
        # K.V ==> (B, N, T, D) * (B, N, D, T)  -> (B, N, T, T)
        attn = q @ k.transpose(-2, -1)

        # scaled dot product ==> divide by square_root of head_dim
        attn = attn * (1.0 / math.sqrt(self.head_dim)) # (B, N, T, T)

        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # multiply ==> (attn * v) ==> (B, N, T, T) * (B, N, T, D)  ->  (B, N, T, D)
        out = attn @ v  # (B, N, T, D)
        
        # reshape to (B, T, N, D) and then convert to shape (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
          
        # apply last projection layer
        out = self.c_proj(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, config: CharGPTConfig, multi_head: bool = False):
        super().__init__()
        # pre-norm transformer block
        self.ln_1 = LayerNormalization(config.emb_dim)
        self.attention = MultiHeadAttention(config) if multi_head else Attention(config)
        self.ln_2 = LayerNormalization(config.emb_dim)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor):
        # residual connections around attention and feed-forward
        x = x + self.attention(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


# create a CharGPT class which extends torch.nn module and implements a small transformer
class CharGPT(nn.Module):
    def __init__(self, vocab_size: int, config: CharGPTConfig = CharGPTConfig()):
        super().__init__()
        self.config = config
        self.config.vocab_size = vocab_size

        # token embedding + positional encoding
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.emb_dim)
        self.positional_encoding = PositionalEncoding(self.config.emb_dim, self.config.block_size)

        # stack of transformer blocks
        self.h = nn.ModuleList([AttentionBlock(self.config, multi_head=True) for _ in range(self.config.n_layers)])
        self.ln_f = LayerNormalization(self.config.emb_dim)

        # output head projects back to vocabulary logits
        self.lm_head = nn.Linear(self.config.emb_dim, self.config.vocab_size)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        # x: (B, T)

        # `B` is a number of batches getting processed in parallel
        # `T` is the list of tokens in a batch, it is the block_size/context length
        # 'C' is the emb_dim (embedding dimension)
        x = self.token_embedding(x)  # (B, T, C) :: (B, block_size, emb_dim)
        x = self.positional_encoding(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, ids: torch.Tensor, max_len: int = 1000):
        # Autoregressive generation that appends tokens to the input ids.
        # Keeps execution on the same device and uses evaluation mode.
        self.eval()
        device = next(self.parameters()).device
        ids = ids.to(device)
        with torch.no_grad():
            for _ in range(max_len):
                logits, _ = self(ids)
                logits = logits[:, -1, :]  # (B, vocab)
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
                ids = torch.cat((ids, next_idx), dim=1)
        return ids


def main():
    # entry-point training loop preserved from original file with safer device selection and sampler wiring.
    data_path = Path(__file__).parents[2] / 'data' / 'shakespeare.txt'
    if not data_path.exists():
        raise FileNotFoundError(f"expected data at {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # build character vocabulary
    chrs = sorted(list(set(text)))
    char_to_id = {ch: idx for idx, ch in enumerate(chrs)}
    id_to_char = {idx: ch for ch, idx in char_to_id.items()}

    encoder = lambda s: [char_to_id[c] for c in s]
    decoder = lambda ids: "".join([id_to_char[i] for i in ids])

    # small subset for quick runs
    text = text[:20000]
    data = torch.tensor(encoder(text), dtype=torch.long)

    # train / validation split
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    config = CharGPTConfig()
    # datasets
    train_dataset = CharDataset(train_data, config.block_size)
    val_dataset = CharDataset(test_data, config.block_size)

    # samplers and dataloaders: ensure RandomSampler is actually used by DataLoader
    train_sampler = RandomSampler(train_dataset, replacement=False, num_samples=10 * config.batch_size)
    val_sampler = RandomSampler(val_dataset, replacement=False, num_samples=10 * config.batch_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)

    # safe device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("using device %s", device)

    # create model and optimizer
    model = CharGPT(len(chrs), config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(sum(train_losses) / len(train_losses))

        # Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, val_loss = model(xb, yb)
                val_losses.append(val_loss.item())

        avg_val_loss = float(sum(val_losses) / len(val_losses))
        logger.info("epoch %d: train loss %.4f, val loss %.4f", epoch, avg_train_loss, avg_val_loss)

    logger.info("training completed")

    # generate from a start token (ensure start token exists in vocab)
    start_char = 'H' if 'H' in char_to_id else chrs[0]
    start_token = torch.tensor([[char_to_id[start_char]]], dtype=torch.long).to(device)
    generated_ids = model.generate(start_token, max_len=200)
    print("Generated text:", decoder(generated_ids[0].cpu().tolist()))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
