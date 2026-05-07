import torch
import torch.nn as nn
from generate import PAD as _PAD, SOS as _SOS, VOCAB_SIZE as _VOCAB_SIZE


class BasicRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Linear(input_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h):
        return torch.tanh(self.W(x) + self.U(h))


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wc = nn.Linear(input_size, hidden_size, bias=False)
        self.Uc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wi = nn.Linear(input_size, hidden_size, bias=False)
        self.Ui = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wf = nn.Linear(input_size, hidden_size, bias=False)
        self.Uf = nn.Linear(hidden_size, hidden_size, bias=True)
        self.Wo = nn.Linear(input_size, hidden_size, bias=False)
        self.Uo = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, h, c):
        c_tilde = torch.tanh(self.Wc(x) + self.Uc(h))
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class RNNSeq2Seq(nn.Module):
    def __init__(self, vocab_size=_VOCAB_SIZE, embed_size=32, hidden_size=128,
                 pad_idx=_PAD, sos_idx=_SOS):
        super().__init__()
        self.hidden_size = hidden_size
        self.sos_idx = sos_idx
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.enc_cell = BasicRNNCell(embed_size, hidden_size)
        self.dec_cell = BasicRNNCell(embed_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def encode(self, src, lengths):
        B, T = src.shape
        h = torch.zeros(B, self.hidden_size, device=src.device)
        emb = self.embed(src)
        for t in range(T):
            h = self.enc_cell(emb[:, t], h)
            # freeze hidden state once past a sample's real length
            mask = (t < lengths).float().unsqueeze(1)
            h = h * mask + h.detach() * (1 - mask)
        return h

    def decode(self, h, tgt):
        B, M = tgt.shape
        token = torch.full((B,), self.sos_idx, dtype=torch.long, device=tgt.device)
        logits = []
        for _ in range(M):
            x = self.embed(token)
            h = self.dec_cell(x, h)
            logit = self.out_proj(h)
            logits.append(logit)
            token = logit.argmax(dim=-1)
        return torch.stack(logits, dim=1)

    def forward(self, src, tgt, lengths):
        h = self.encode(src, lengths)
        return self.decode(h, tgt)

    @torch.no_grad()
    def predict(self, src, lengths, max_len=8):
        B = src.size(0)
        h = self.encode(src, lengths)
        token = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        preds = []
        for _ in range(max_len):
            x = self.embed(token)
            h = self.dec_cell(x, h)
            logit = self.out_proj(h)
            token = logit.argmax(dim=-1)
            preds.append(token)
        return torch.stack(preds, dim=1)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size=_VOCAB_SIZE, embed_size=32, hidden_size=128,
                 pad_idx=_PAD, sos_idx=_SOS):
        super().__init__()
        self.hidden_size = hidden_size
        self.sos_idx = sos_idx
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.enc_cell = LSTMCell(embed_size, hidden_size)
        self.dec_cell = LSTMCell(embed_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def encode(self, src, lengths):
        B, T = src.shape
        h = torch.zeros(B, self.hidden_size, device=src.device)
        c = torch.zeros(B, self.hidden_size, device=src.device)
        emb = self.embed(src)
        for t in range(T):
            h_new, c_new = self.enc_cell(emb[:, t], h, c)
            mask = (t < lengths).float().unsqueeze(1)
            h = h_new * mask + h * (1 - mask)
            c = c_new * mask + c * (1 - mask)
        return h, c

    def decode(self, h, c, tgt):
        B, M = tgt.shape
        token = torch.full((B,), self.sos_idx, dtype=torch.long, device=tgt.device)
        logits = []
        for _ in range(M):
            x = self.embed(token)
            h, c = self.dec_cell(x, h, c)
            logit = self.out_proj(h)
            logits.append(logit)
            token = logit.argmax(dim=-1)
        return torch.stack(logits, dim=1)

    def forward(self, src, tgt, lengths):
        h, c = self.encode(src, lengths)
        return self.decode(h, c, tgt)

    @torch.no_grad()
    def predict(self, src, lengths, max_len=8):
        B = src.size(0)
        h, c = self.encode(src, lengths)
        token = torch.full((B,), self.sos_idx, dtype=torch.long, device=src.device)
        preds = []
        for _ in range(max_len):
            x = self.embed(token)
            h, c = self.dec_cell(x, h, c)
            logit = self.out_proj(h)
            token = logit.argmax(dim=-1)
            preds.append(token)
        return torch.stack(preds, dim=1)
