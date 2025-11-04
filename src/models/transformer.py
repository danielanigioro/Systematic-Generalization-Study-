# src/models/transformer.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

    def forward(self, x):
        # x: (B, L, d)
        L = x.size(1)
        return x + self.pe[:, :L, :]

def _subsequent_mask(sz):
    # (sz, sz) bool mask with True for invalid positions (upper triangle)
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)

class Seq2SeqTransformer(nn.Module):
    """
    Tiny encoder-decoder Transformer for tokens->actions.
    - Token side uses tok_vocab; action side uses act_vocab (with <bos>/<eos>).
    """
    def __init__(self, tok_vocab, act_vocab,
                 d_model=256, nhead=4, num_layers=3, dim_ff=512, dropout=0.1):
        super().__init__()
        self.tok_vocab = tok_vocab
        self.act_vocab = act_vocab
        self.src_vocab_size = len(tok_vocab)
        self.tgt_vocab_size = len(act_vocab)

        self.src_emb = nn.Embedding(self.src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        self.out = nn.Linear(d_model, self.tgt_vocab_size)

    def encode(self, src, src_key_padding_mask=None):
        x = self.pos(self.src_emb(src))           # (B, S, d)
        mem = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return mem                                 # (B, S, d)

    def decode(self, tgt_inp, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        y = self.pos(self.tgt_emb(tgt_inp))       # (B, T, d)
        dec = self.decoder(y, memory,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return self.out(dec)                       # (B, T, Vtgt)

    def forward(self, x, x_lens, y=None, tf=0.0, max_steps=None):
        """
        x: (B, S) token ids; y: (B, T) action ids incl <bos> ... <eos>
        If y is None → greedy decode up to max_steps.
        """
        B, S = x.size()
        device = x.device
        pad_id = 0

        # padding masks: True for pads
        src_pad_mask = (x == pad_id)  # (B, S)

        memory = self.encode(x, src_key_padding_mask=src_pad_mask)

        if y is not None:
            # Teacher-forcing path: shift right (input) vs gold (target)
            tgt_inp = y[:, :-1]                     # drop last
            tgt_pad_mask = (tgt_inp == pad_id)
            T = tgt_inp.size(1)
            tgt_mask = _subsequent_mask(T).to(device)
            logits = self.decode(tgt_inp, memory,
                                 tgt_mask=tgt_mask,
                                 tgt_key_padding_mask=tgt_pad_mask,
                                 memory_key_padding_mask=src_pad_mask)
            # Pad logits to (B, T_out= y_len-1)
            return logits
        else:
            # Greedy decode
            assert max_steps is not None
            bos_id, eos_id = 1, 2
            ys = torch.full((B,1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            logits_list = []
            for t in range(max_steps-1):
                tgt_mask = _subsequent_mask(ys.size(1)).to(device)
                tgt_pad_mask = (ys == pad_id)
                step_logits = self.decode(ys, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_pad_mask,
                                          memory_key_padding_mask=src_pad_mask)
                logits_list.append(step_logits[:, -1:, :])  # last step
                next_token = step_logits[:, -1, :].argmax(-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)
                if finished.all():
                    break
            # cat time
            if logits_list:
                return torch.cat(logits_list, dim=1)  # (B, T_gen, V)
            else:
                # degenerate case (max_steps <= 1)
                V = self.tgt_vocab_size
                return torch.zeros(B, 0, V, device=device)

