import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.causal_attn = nn.MultiheadAttention(
            config.n_embd,
            config.n_head,
            batch_first=True,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(self, x):
        _, T, _ = x.size()
        y = self.causal_attn(
            x,
            x,
            x,
            is_causal=True,
            need_weights=False,
            attn_mask=self.mask[:T, :T].to(x.device),  # broadcasted
        )[0]
        # y = x + y
        # is this residual connection? if yes, why is it needed here
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttn(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class DOTConfig:
    block_size: int = 1024
    # should i change any other config params if i changd the vocab_size
    vocab_size: int = 16768
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class DOT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # refactor, how is c_proj named, inited inside multihead proj | why apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print(
            "Number of parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _init_weights(self, module):
        # understand 0.2 init
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.2)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(t, device=device).expand(b, t)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # understand why are we generating x of shape B, T, C(embd_size) if we only need last token while inference: this is sensible during training i.e., to train

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, temp=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_nxt), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of parameters with decay: {num_decay_params}")
        print(f"Number of parameters without decay: {num_no_decay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print("Using fused optimizer: ", use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # understand this code
        N = sum(p.numel() for p in self.parameters())-self.transformer.wpe.weight.numel()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T  
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achived = flops_per_iter / (1.0*dt)
        flops_promised = 3e12 # change according to A100 GPU available
        mfu = flops_achived/flops_promised
        return mfu 
