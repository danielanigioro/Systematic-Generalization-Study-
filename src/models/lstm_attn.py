import torch, torch.nn as nn, torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, d_enc, d_dec, d_att=256):
        super().__init__()
        self.Wq = nn.Linear(d_dec, d_att)
        self.Wk = nn.Linear(d_enc, d_att)
        self.v  = nn.Linear(d_att, 1, bias=False)
    def forward(self, h_enc, h_dec, mask):
        # h_enc: (B,T_e,D_e), h_dec: (B,1,D_d)
        q = self.Wq(h_dec)              # (B,1,d)
        k = self.Wk(h_enc)              # (B,T_e,d)
        scores = self.v(torch.tanh(q + k))  # (B,T_e,1)
        scores = scores.squeeze(-1)     # (B,T_e)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=1).unsqueeze(1) # (B,1,T_e)
        ctx = w @ h_enc                 # (B,1,D_e)
        return ctx.squeeze(1), w.squeeze(1)  # (B,D_e), (B,T_e)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_in, vocab_out, emb=128, hid=256, layers=2, dropout=0.1):
        super().__init__()
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.Ein  = nn.Embedding(len(vocab_in), emb, padding_idx=self.pad_id)
        self.Eout = nn.Embedding(len(vocab_out), emb, padding_idx=self.pad_id)
        self.enc  = nn.LSTM(emb, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.dec  = nn.LSTM(emb + hid, hid, num_layers=layers, batch_first=True, dropout=dropout)
        self.att  = AdditiveAttention(hid, hid)
        self.proj = nn.Linear(hid, len(vocab_out))

    def encode(self, x, x_lens):
        packed = nn.utils.rnn.pack_padded_sequence(self.Ein(x), x_lens.cpu(), batch_first=True, enforce_sorted=False)
        h,_ = self.enc(packed)
        h,_ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return h  # (B,T,hid)

    def forward(self, x, x_lens, y=None, tf=0.5, max_steps=64):
        B = x.size(0)
        h_enc = self.encode(x, x_lens)
        mask = (x != self.pad_id)
        dec_in = torch.full((B,1), self.bos_id, dtype=torch.long, device=x.device)
        h = None
        logits = []
        for t in range(max_steps):
            ctx,_ = self.att(h_enc, h[0][-1].unsqueeze(1) if h else torch.zeros(B,1,h_enc.size(-1),device=x.device), mask)
            emb = self.Eout(dec_in).squeeze(1)         # (B,emb)
            dec_in_vec = torch.cat([emb, ctx], dim=-1) # (B,emb+hid)
            out, h = self.dec(dec_in_vec.unsqueeze(1), h)  # out: (B,1,hid)
            step_logits = self.proj(out.squeeze(1))    # (B,|V_out|)
            logits.append(step_logits.unsqueeze(1))
            # teacher forcing
            if y is not None and torch.rand(1).item() < tf and t < y.size(1)-1:
                dec_in = y[:,t+1:t+2]
            else:
                dec_in = step_logits.argmax(-1, keepdim=True)
        return torch.cat(logits, dim=1)  # (B,T*,|V_out|)
