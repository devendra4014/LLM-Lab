import torch

from LLM_HandsOn.train.CharGPT import CharGPT
import torch.nn.functional as F


def generate(m: CharGPT, start_token, max_len=1000):
    for i in range(max_len):
        logits, loss = m(start_token)  # (batch, token , vocab_size)

        # select last token prediction for a batch of vocab size
        logits = logits[:, -1, :]  # token , vocab_size ==> (1, vocab_size)

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        start_token = torch.cat((start_token, idx_next), dim=1)

    return start_token


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CharGPT(len(chrs))  # create model instance
model.load_state_dict(torch.load("../train/models/char_gpt_trained.pth", weights_only=False))
model.to(device)
model.eval()

# Generation example
start_token = torch.tensor([[char_to_id['H']]]).to(device)
generated_text = generate(model, start_token)

print("Generated text:", decoder(generated_text[0].tolist()))
