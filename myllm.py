import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = 4

    def build_vocab(self, texts, min_freq=2):
        all_chars = []
        for text in texts:
            cleaned_text = re.sub(r'[^\u0900-\u097F\s.,!?।]', '', text)
            all_chars.extend(list(cleaned_text))
        char_counts = Counter(all_chars)
        idx = 4
        for char, count in char_counts.items():
            if count >= min_freq and char not in self.vocab:
                self.vocab[char] = idx
                self.idx_to_token[idx] = char
                idx += 1
        self.vocab_size = len(self.vocab)
        print(f"🔤 Hindi Vocabulary Size: {self.vocab_size}")

    def encode(self, text):
        tokens = [self.vocab['<bos>']]
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab['<unk>']))
        tokens.append(self.vocab['<eos>'])
        return tokens

    def decode(self, token_ids):
        result = ""
        for token_id in token_ids:
            if token_id in [self.vocab['<bos>'], self.vocab['<pad>']]:
                continue
            elif token_id == self.vocab['<eos>']:
                break
            result += self.idx_to_token.get(token_id, '')
        return result

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                continue
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.vocab['<pad>']] * pad_len
            target_ids += [tokenizer.vocab['<pad>']] * pad_len
            self.data.append((torch.tensor(input_ids), torch.tensor(target_ids)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=4, d_ff=512, max_length=64):
        super().__init__()
        self.max_length = max_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.head(x)

def generate_text(model, tokenizer, prompt="मैं", max_length=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated = input_ids.tolist()[0]
    for _ in range(max_length):
        if input_ids.shape[1] >= model.max_length:
            break
        with torch.no_grad():
            logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == tokenizer.vocab['<eos>']:
            break
        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(generated)

def train_model(model, dataloader, optimizer, criterion, epochs=100):
    model.train()
    all_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(dataloader)
        all_losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")
    return all_losses

# ---------------- Run Pipeline ----------------
sample_texts_hi = [
    "आज मौसम बहुत अच्छा है।", "मैं स्कूल जा रहा हूँ।", "तुम क्या कर रहे हो?",
    "मुझे चाय पसंद है।", "यह किताब बहुत रोचक है।", "हम पार्क में खेलते हैं।",
    "कल मेरी परीक्षा है।", "माँ खाना बना रही हैं।", "सड़क पर बहुत ट्रैफिक है।",
    "मैं संगीत सुनना पसंद करता हूँ।", "तुम्हारा नाम क्या है?", "भारत एक सुंदर देश है।",
    "हमने फिल्म देखी।", "बिल्ली कुर्सी पर सो रही है।", "खिड़की से हवा आ रही है।",
    "पढ़ाई समय पर करनी चाहिए।", "पापा दफ्तर जा चुके हैं।", "छुट्टी के दिन मैं घूमने गया।",
    "बच्चे मैदान में दौड़ रहे हैं।", "आज रात को चाँदनी बहुत सुंदर है।"
]

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(sample_texts_hi)
dataset = TextDataset(sample_texts_hi, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleLLM(vocab_size=tokenizer.vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])

print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
losses = train_model(model, dataloader, optimizer, criterion, epochs=50)

# ✅ Plot training loss (non-blocking)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show(block=False)
plt.pause(2)
plt.close()

# Generate a sample
print("\n👉 Generating Hindi text...")
print(generate_text(model, tokenizer, prompt="मैं स्कूल", max_length=40, temperature=0.8))
