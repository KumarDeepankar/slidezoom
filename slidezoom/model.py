"""SlideZoom: a frozen GPT-2 that reads a slide coarse-to-fine through progressive-zoom cross-attention.

    GPT-2 block 3  ──► x-attn over regions      (~4 mm)        "where should I look?"
    GPT-2 block 6  ──► x-attn over sub-regions  (~1 mm)   ┐
    GPT-2 block 9  ──► x-attn over tiles                  ├ each guided by the previous block's attention
    GPT-2 block 12 ──► x-attn over tiles                  ┘

Zoom prior. For text token t, block l's attention logits over cells c get a bias from what block l-1
(same token t) attended to at the coarser level:

    logits_l(t, c) = q_t · k_c / sqrt(d_h)  +  beta_l · log a_{l-1}(t, ancestor(c))

beta_l starts at 0, so at initialisation this is exactly a multi-scale Flamingo model; training decides
how strongly to zoom. At inference, `set_topk` keeps only the children of the top-k cells of the
previous block (per text token), so the model can read a whole slide while scoring few tiles.

Modes (same code, so comparisons are fair):
    zoom        levels (0, 1, 2, 2), learned beta          <- the proposed model
    multiscale  levels (0, 1, 2, 2), beta fixed at 0      <- B2: hierarchy without the zoom prior
    flat        levels (2, 2, 2, 2), no prior             <- B0 / B1: SlideTalk's cross-attention
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

XATTN_AFTER = (2, 5, 8, 11)  # GPT-2 block indices (0-based): cross-attention after blocks 3, 6, 9, 12
SCHEDULES = {"zoom": (0, 1, 2, 2), "multiscale": (0, 1, 2, 2), "flat": (2, 2, 2, 2)}
N_LEVELS = 3


class ZoomCrossAttention(nn.Module):
    """Gated cross-attention whose logits can be biased by a prior over the cells it reads.

    x       (S*Q, T, d)  text states; slide s owns rows s*Q .. s*Q+Q-1   -> queries
    mem     (S, N, d)    memory of one level of the slide                -> keys and values
    mask    (S, N)       True for real cells
    prior   (S, Q*T, N)  probability each text token gives each cell's ancestor (None: no prior)
    allowed (S, Q*T, N)  top-k pruning: cells this token may read (None: all)
    returns x + tanh(gate) * W_o(read), and attention (S, heads, Q*T, N)
    """

    def __init__(self, d: int = 768, heads: int = 12, learn_beta: bool = True):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.ln = nn.LayerNorm(d)
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d, d) for _ in range(4))
        self.gate = nn.Parameter(torch.zeros(1))  # tanh(0) = 0: starts as plain GPT-2 (Flamingo)
        if learn_beta:
            self.beta = nn.Parameter(torch.zeros(1))  # 0: starts with no zoom
        else:
            self.register_buffer("beta", torch.zeros(1))

    def kv(self, mem):
        S, N, _ = mem.shape
        k = self.W_k(mem).view(S, N, self.h, self.dh).transpose(1, 2)  # (S, h, N, dh)
        v = self.W_v(mem).view(S, N, self.h, self.dh).transpose(1, 2)
        return k, v

    def forward(self, x, mem, mask, prior=None, allowed=None, kv=None):
        B, T, d = x.shape
        S = mem.shape[0]
        assert B % S == 0, (B, S)
        QT = (B // S) * T
        q = self.W_q(self.ln(x)).reshape(S, QT, self.h, self.dh).transpose(1, 2)  # (S, h, QT, dh)
        k, v = kv if kv is not None else self.kv(mem)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                     # (S, h, QT, N)
        if prior is not None:
            scores = scores + self.beta * torch.log(prior.clamp_min(1e-6))[:, None]
        keep = mask[:, None, :].expand(S, QT, -1)                                 # (S, QT, N)
        if allowed is not None:
            pruned = keep & allowed
            keep = torch.where(pruned.any(-1, keepdim=True), pruned, keep)       # never leave a row empty
        scores = scores.masked_fill(~keep[:, None], torch.finfo(scores.dtype).min)
        attn = scores.softmax(-1)
        read = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return x + torch.tanh(self.gate) * self.W_o(read), attn


class SlideZoomer(nn.Module):
    def __init__(self, mode: str = "zoom", lm_name: str = "gpt2", tile_dim: int = 768, mem_dropout: float = 0.0):
        super().__init__()
        self.mode, self.levels = mode, SCHEDULES[mode]
        self.use_prior = mode != "flat"
        self.tok = GPT2TokenizerFast.from_pretrained(lm_name)
        self.tok.pad_token = self.tok.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained(lm_name)
        self.lm.requires_grad_(False)
        d, heads = self.lm.config.n_embd, self.lm.config.n_head
        self.proj = nn.Sequential(nn.LayerNorm(tile_dim), nn.Linear(tile_dim, d))  # tile space -> word space
        self.level_emb = nn.Embedding(N_LEVELS, d)                                  # tells blocks the scale
        nn.init.normal_(self.level_emb.weight, std=0.02)
        self.mem_drop = nn.Dropout(mem_dropout)
        self.xattn = nn.ModuleList([ZoomCrossAttention(d, heads, learn_beta=(mode == "zoom" and j > 0))
                                    for j in range(len(XATTN_AFTER))])
        self.topk: list[int | None] = [None] * len(XATTN_AFTER)
        self.keep_attn = False  # store each block's attention (for maps); off in training
        self._slide = None
        for j, i in enumerate(XATTN_AFTER):
            self.lm.transformer.h[i].register_forward_hook(self._hook(j))
        self.train(False)

    # ------------------------------------------------------------------ plumbing
    def train(self, mode: bool = True):
        super().train(mode)
        self.lm.eval()  # GPT-2 is frozen: keep its dropout off
        return self

    @property
    def device(self):
        return self.proj[1].weight.device

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}

    def load_trainable(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("lm.") for k in missing), (missing, unexpected)

    def gates(self) -> list[float]:
        return [torch.tanh(b.gate).item() for b in self.xattn]

    def betas(self) -> list[float]:
        return [b.beta.item() for b in self.xattn]

    def set_topk(self, ks):
        """ks: one k per zoom-in step (e.g. (2, 8): 2 regions, then 8 sub-regions), or None to read everything."""
        self.topk = [None] * len(XATTN_AFTER)
        if ks is None:
            return
        steps = [j for j in range(1, len(self.levels)) if self.levels[j] > self.levels[j - 1]]
        assert self.use_prior and len(ks) == len(steps), f"{self.mode} has {len(steps)} zoom-in steps"
        for j, k in zip(steps, ks):
            self.topk[j] = k

    # ------------------------------------------------------------------ slide memory
    def set_slide(self, batch: dict | None):
        """batch from data.collate (per level feats / mask / parent), or None for plain GPT-2."""
        if batch is None:
            self._slide = None
            return
        mem = {l: self.mem_drop(self.proj(batch["feats"][l]) + self.level_emb.weight[l]) for l in set(self.levels)}
        self._slide = dict(mem=mem, mask=batch["mask"], parent=batch["parent"], kv={}, anc={})
        self.attn_log, self.attended = {}, None

    def _ancestor(self, fine: int, coarse: int) -> torch.Tensor:
        """(S, N_fine) index of each fine cell's ancestor at the coarse level."""
        key = (fine, coarse)
        if key not in self._slide["anc"]:
            a = self._slide["parent"][fine]
            for l in range(fine - 1, coarse, -1):
                a = self._slide["parent"][l].gather(1, a)
            self._slide["anc"][key] = a
        return self._slide["anc"][key]

    def _hook(self, j):
        def run(_block, _inputs, out):
            if self._slide is None:  # no slide: plain GPT-2
                return None
            h = out[0] if isinstance(out, tuple) else out
            h = self._xattn(j, h)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return run

    def _xattn(self, j, h):
        sl, lvl = self._slide, self.levels[j]
        mem, mask = sl["mem"][lvl], sl["mask"][lvl]
        kv = None
        if not torch.is_grad_enabled():  # memory is fixed while answering: project it once per slide
            kv = sl["kv"].get(j) or sl["kv"].setdefault(j, self.xattn[j].kv(mem))
        prior = allowed = None
        if self.use_prior and j > 0:
            prev_lvl, A, prev_allowed = self.levels[j - 1], sl["prev_attn"], sl["prev_allowed"]
            if lvl == prev_lvl:
                prior, allowed = A, prev_allowed
            else:
                idx = self._ancestor(lvl, prev_lvl)[:, None, :].expand(-1, A.shape[1], -1)  # (S, QT, N)
                prior = A.gather(2, idx)
                if self.topk[j] is not None:
                    top = A.topk(min(self.topk[j], A.shape[-1]), -1).indices
                    keep_parent = torch.zeros_like(A, dtype=torch.bool).scatter(2, top, True)
                    if prev_allowed is not None:
                        keep_parent &= prev_allowed
                    allowed = keep_parent.gather(2, idx)
        h, attn = self.xattn[j](h, mem, mask, prior, allowed, kv)
        sl["prev_attn"] = attn.mean(1)  # (S, QT, N): average over heads
        sl["prev_allowed"] = allowed
        if lvl == N_LEVELS - 1:  # how many tiles each text token could read
            n = mask[:, None, :] if allowed is None else (allowed & mask[:, None, :])
            self.attended = n.sum(-1).float().mean().item()
        if self.keep_attn:
            self.attn_log[j] = attn.detach()
        return h

    @staticmethod
    def prompt(question: str) -> str:
        return f"Question: {question}\nAnswer:"

    # ------------------------------------------------------------------ training
    def loss(self, slide, questions, answers, reduce: bool = True):
        """Next-word loss on the answer words only. Questions are slide-major (see data.make_batch)."""
        self.set_slide(slide)
        ids, labels = [], []
        for q, a in zip(questions, answers):
            p = self.tok(self.prompt(q)).input_ids
            t = self.tok(" " + a).input_ids + [self.tok.eos_token_id]
            ids.append(p + t)
            labels.append([-100] * len(p) + t)
        L = max(map(len, ids))
        pad = self.tok.eos_token_id
        input_ids = torch.tensor([s + [pad] * (L - len(s)) for s in ids], device=self.device)
        target = torch.tensor([s + [-100] * (L - len(s)) for s in labels], device=self.device)
        attention = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids], device=self.device)
        out = self.lm(input_ids=input_ids, attention_mask=attention, labels=target if reduce else None)
        if reduce:
            return out.loss
        logp = out.logits[:, :-1].log_softmax(-1)
        tgt = target[:, 1:]
        nll = -logp.gather(2, tgt.clamp_min(0)[..., None])[..., 0] * (tgt != -100)
        return nll.sum(1) / (tgt != -100).sum(1)  # (B,) mean NLL per answer

    # ------------------------------------------------------------------ talking
    @torch.no_grad()
    def answer(self, slide: dict | None, question: str, max_new: int = 24) -> str:
        """Greedy decoding about one slide (a collate of one hierarchy), or None for no slide."""
        self.set_slide(slide)
        ids = self.tok(self.prompt(question), return_tensors="pt").input_ids.to(self.device)
        start, past, step = ids.shape[1], None, ids
        for _ in range(max_new):
            out = self.lm(input_ids=step, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[0, -1].argmax().view(1, 1)
            if nxt.item() == self.tok.eos_token_id:
                break
            ids, step = torch.cat([ids, nxt], 1), nxt
        return self.tok.decode(ids[0, start:]).strip()

    @torch.no_grad()
    def attention_map(self, slide: dict, question: str) -> dict[int, np.ndarray]:
        """Where the last prompt token looked at each level: {level: (N_level,)}, averaged over heads/blocks."""
        self.keep_attn = True
        try:
            self.set_slide(slide)
            self.lm(input_ids=self.tok(self.prompt(question), return_tensors="pt").input_ids.to(self.device))
            maps = {}
            for j, a in self.attn_log.items():
                maps.setdefault(self.levels[j], []).append(a[0, :, -1].mean(0))
            return {l: torch.stack(v).mean(0).float().cpu().numpy() for l, v in maps.items()}
        finally:
            self.keep_attn = False
