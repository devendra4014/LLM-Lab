import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ Model Definition ------------------

class BigramLanguageModel(nn.Module):
    """Bigram Language Model using a single embedding lookup table."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_length):
        for _ in range(max_length):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ------------------ Utility Functions ------------------

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def eval_loss():
    out = dict()
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for i in range(eval_interval):
            X, Y = get_batch(split)
            _, e_loss = model(X, Y)
            losses[i] = e_loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ------------------ Main Execution ------------------

if __name__ == '__main__':
    # Load and encode data
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chrs = sorted(list(set(text)))
    print(f"Number of unique characters: {len(chrs)}")

    stoi = {char: i for i, char in enumerate(chrs)}
    itos = {i: char for i, char in enumerate(chrs)}
    encoder = lambda s: [stoi[c] for c in s]
    decoder = lambda l: ''.join([itos[i] for i in l])

    n = len(text)
    data = torch.tensor(encoder(text), dtype=torch.long)
    n_train = int(0.9 * n)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # Hyperparameters
    block_size = 8
    batch_size = 32
    vocab_size = len(chrs)
    learning_rate = 1e-3
    max_iters = 5000
    eval_interval = 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model and optimizer
    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for i in range(max_iters):
        if i % eval_interval == 0:
            losses = eval_loss()
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generation example
    start_token = torch.tensor([[stoi['H']]]).to(device)
    generated_text = model.generate(start_token, max_length=50)
    print("Generated text:", decoder(generated_text[0].tolist()))
