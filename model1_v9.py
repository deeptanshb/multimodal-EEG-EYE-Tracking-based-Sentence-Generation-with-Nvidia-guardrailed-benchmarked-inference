import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


# ══════════════════════════════════════════════════════════════════
# LoRA
# ══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear
        self.scale  = alpha / rank
        if hasattr(linear, 'in_features'):
            in_f, out_f = linear.in_features, linear.out_features
        else:
            in_f, out_f = linear.weight.shape[0], linear.weight.shape[1]
        self.lora_A = nn.Linear(in_f, rank,  bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.scale * self.lora_B(self.lora_A(x))


def _apply_lora(gpt2, block_indices, rank=8, alpha=16.0):
    device = next(gpt2.parameters()).device
    for i in block_indices:
        blk = gpt2.transformer.h[i]
        blk.attn.c_attn = LoRALinear(blk.attn.c_attn, rank, alpha).to(device)
        blk.attn.c_proj = LoRALinear(blk.attn.c_proj, rank, alpha).to(device)


# ══════════════════════════════════════════════════════════════════
# Region indices
# ══════════════════════════════════════════════════════════════════

REGION_INDICES = {
    "left_temporal":           [16, 21, 22, 23],
    "left_parietal":           [ 1,  7,  8,  9, 14, 19],
    "left_parieto_occipital":  [ 0,  3,  4, 11, 12, 17],
    "central_parietal":        [ 2,  6, 15],
    "right_parietal":          [ 5, 10, 20],
    "right_parieto_occipital": [13, 18],
}
REGION_NAMES = list(REGION_INDICES.keys())


# ══════════════════════════════════════════════════════════════════
# FIX-1: Hierarchical Temporal Pooling
# Replaces the flat pool_attn Linear(D,1) that collapsed to 1/256.
# Two-level softmax: 32-way local within each window + 8-way across
# windows. Gradient signal is 8x more concentrated → model learns
# to peak rather than staying uniform.
# ══════════════════════════════════════════════════════════════════

class HierarchicalTemporalPooling(nn.Module):
    """
    Two-level attention pooling over T timesteps.

    Level 1 (local):   split T into n_segments windows of seg_len each.
                       learn which timesteps matter within each window.
    Level 2 (segment): learn which windows matter across the sequence.
    Residual + LayerNorm on the pooled output.

    Prevents 1/T collapse by operating on much smaller softmax
    denominators at each level (32-way and 8-way vs 256-way).
    """

    def __init__(self, dim: int, n_segments: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_segments = n_segments
        self.local_attn = nn.Linear(dim, 1)
        self.seg_attn   = nn.Linear(dim, 1)
        self.seg_proj   = nn.Linear(dim, dim)
        self.norm       = nn.LayerNorm(dim)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x       : (B, T, D)
        returns : emb (B, D),  local_w (B, T_trim, 1),  seg_w (B, n_segments, 1)
        """
        B, T, D  = x.shape
        seg_len  = T // self.n_segments
        T_trim   = seg_len * self.n_segments

        x_seg    = x[:, :T_trim, :].view(B, self.n_segments, seg_len, D)
        local_w  = F.softmax(self.local_attn(x_seg), dim=2)   # (B, S, seg_len, 1)
        seg_emb  = (x_seg * local_w).sum(dim=2)               # (B, S, D)

        seg_w    = F.softmax(self.seg_attn(seg_emb), dim=1)   # (B, S, 1)
        out      = (seg_emb * seg_w).sum(dim=1)               # (B, D)
        out      = self.norm(out + self.drop(self.seg_proj(out)))

        return out, local_w.view(B, T_trim, 1), seg_w


# ══════════════════════════════════════════════════════════════════
# Sub-modules
# ══════════════════════════════════════════════════════════════════

class RegionEncoderV9(nn.Module):
    """
    Drop-in replacement for RegionEncoder (V8).
    Replaces flat pool_attn with HierarchicalTemporalPooling.
    Returns (emb, (local_w, seg_w)) — EEGEncoder unpacks accordingly.
    """

    def __init__(self, n_channels: int, region_dim: int = 384,
                 n_heads: int = 4, dropout: float = 0.3, n_segments: int = 8):
        super().__init__()
        self.gru = nn.GRU(input_size=n_channels, hidden_size=region_dim,
                          num_layers=1, batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=region_dim, nhead=n_heads, dim_feedforward=region_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.htp = HierarchicalTemporalPooling(region_dim,
                                               n_segments=n_segments,
                                               dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        x       : (B, T, n_channels)
        returns : emb (B, region_dim),  (local_w (B, T, 1), seg_w (B, S, 1))
        """
        gru_out = self.gru(x)[0]
        t_out   = self.transformer(gru_out)
        emb, local_w, seg_w = self.htp(t_out)
        return emb, (local_w, seg_w)


class EEGEncoder(nn.Module):
    def __init__(self, hidden_dim, region_dim=384, n_heads=4, dropout=0.3):
        super().__init__()
        self.region_encoders = nn.ModuleDict({
            name: RegionEncoderV9(len(REGION_INDICES[name]),
                                  region_dim, n_heads, dropout)
            for name in REGION_NAMES
        })
        self.region_proj = nn.Linear(region_dim, hidden_dim)
        for name, idxs in REGION_INDICES.items():
            self.register_buffer(f"_idx_{name}",
                                 torch.tensor(idxs, dtype=torch.long))

    def forward(self, x):
        tokens, attn_weights = [], []
        for name in REGION_NAMES:
            idx = getattr(self, f"_idx_{name}")
            emb, (local_w, seg_w) = self.region_encoders[name](x[:, :, idx])
            tokens.append(emb)
            attn_weights.append((local_w, seg_w))          # tuple per region
        stacked = torch.stack(tokens, dim=1)               # (B, 6, region_dim)
        return self.region_proj(stacked), attn_weights      # (B, 6, H), list[6]


class EyeEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, H), nn.ReLU(), nn.Linear(H, H))
    def forward(self, x): return self.net(x).unsqueeze(1)


class SpectralEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, H), nn.ReLU(), nn.Linear(H, H))
    def forward(self, x): return self.net(x).unsqueeze(1)


class WordSpectralEncoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(400, H), nn.ReLU(), nn.Linear(H, H))
    def forward(self, x):
        if x.dim() == 3: x = x.flatten(1)
        return self.net(x).unsqueeze(1)


class SRConditionAdapter(nn.Module):
    """
    Three separate MLPs — one per reading condition.
    SR gets an extra hidden layer to handle noisy fast-reading signals.
    Applied after EEG fusion as a residual: output = norm(x + adapter(x))
    """
    def __init__(self, H, dropout=0.1):
        super().__init__()
        self.adapter_nr  = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, H))
        self.adapter_tsr = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, H))
        self.adapter_sr  = nn.Sequential(
            nn.Linear(H, H*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H*2, H), nn.GELU(), nn.Linear(H, H))
        self.norm = nn.LayerNorm(H)

    def forward(self, x, condition):
        out = torch.zeros_like(x)
        for cond_id, adapter in [(0, self.adapter_nr),
                                  (1, self.adapter_tsr),
                                  (2, self.adapter_sr)]:
            mask = (condition == cond_id)
            if mask.any():
                out[mask] = adapter(x[mask])
        return self.norm(x + out)


# ══════════════════════════════════════════════════════════════════
# MAIN MODEL  (V9 — HTP encoder, everything else identical to V8)
# ══════════════════════════════════════════════════════════════════

class EEG2TextTransformerV9(nn.Module):

    def __init__(self, gpt_model_name="gpt2", n_heads=4, dropout=0.3,
                 contrast_dim=128, region_dim=384):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        H = self.gpt2.config.hidden_size   # 768

        self.eeg_enc       = EEGEncoder(H, region_dim=region_dim,
                                        n_heads=n_heads, dropout=dropout)
        self.eye_enc       = EyeEncoder(H)
        self.spec_enc      = SpectralEncoder(H)
        self.spec_word_enc = WordSpectralEncoder(H)

        self.fusion        = nn.MultiheadAttention(H, n_heads,
                                                   dropout=dropout, batch_first=True)
        self._fusion_query = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self._fusion_norm  = nn.LayerNorm(H)

        self.enc_proj       = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, H))
        self._enc_proj_norm = nn.LayerNorm(H)

        self.sr_adapter    = SRConditionAdapter(H, dropout=dropout)
        self.contrast_head = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, contrast_dim))

        self.task_prefix   = nn.Parameter(torch.randn(1, 4, H) * 0.02)
        self.condition_emb = nn.Embedding(3, H)
        self._lora_applied = False

        # Stores (local_w, seg_w) per region for visualisation
        self._last_region_attn_w = None

    # ── EEG → single prefix token ─────────────────────────────
    def _encode_eeg(self, eeg, condition=None):
        B = eeg.size(0)
        region_tokens, region_attn_w = self.eeg_enc(eeg)   # (B,6,H), list[6 tuples]
        query    = self._fusion_query.expand(B, -1, -1)
        fused, _ = self.fusion(query, region_tokens, region_tokens)
        fused    = self._fusion_norm(fused + query)
        fused_sq = fused.squeeze(1)
        out      = self._enc_proj_norm(fused_sq + self.enc_proj(fused_sq))
        if condition is not None:
            out  = self.sr_adapter(out, condition)
        self._last_region_attn_w = region_attn_w
        return out.unsqueeze(1)                             # (B, 1, H)

    # ── Stage setup ───────────────────────────────────────────
    def stage_1_setup(self):
        """Freeze GPT2 except lm_head + blocks[10,11]. Train all encoders."""
        for p in self.gpt2.parameters():
            p.requires_grad = False
        for p in self.gpt2.lm_head.parameters():
            p.requires_grad = True
        for i in [10, 11]:
            for p in self.gpt2.transformer.h[i].parameters():
                p.requires_grad = True
        for m in [self.eeg_enc, self.eye_enc, self.spec_enc, self.spec_word_enc,
                  self.enc_proj, self.fusion, self.contrast_head, self.sr_adapter]:
            for p in m.parameters(): p.requires_grad = True
        self._fusion_query.requires_grad = True
        for p in self._fusion_norm.parameters():   p.requires_grad = True
        for p in self._enc_proj_norm.parameters(): p.requires_grad = True
        self.task_prefix.requires_grad          = True
        self.condition_emb.weight.requires_grad = True
        self._print_trainable("Stage 1")

    def stage_2_setup(self, lora_rank=8, lora_alpha=16.0, lora_blocks=None):
        """Apply LoRA to top-2 GPT2 blocks. Call ONCE after loading Stage 1."""
        if lora_blocks is None:
            n = len(self.gpt2.transformer.h)
            lora_blocks = [n-2, n-1]
        if not self._lora_applied:
            _apply_lora(self.gpt2, lora_blocks, lora_rank, lora_alpha)
            self._lora_applied = True
            print(f"[Stage 2] LoRA → blocks {lora_blocks}, rank={lora_rank}")
        for name, p in self.gpt2.named_parameters():
            p.requires_grad = ("lora_A" in name or "lora_B" in name
                               or "lm_head" in name)
        for m in [self.eeg_enc, self.eye_enc, self.spec_enc, self.spec_word_enc,
                  self.enc_proj, self.fusion, self.contrast_head, self.sr_adapter]:
            for p in m.parameters(): p.requires_grad = True
        self._fusion_query.requires_grad = True
        for p in self._fusion_norm.parameters():   p.requires_grad = True
        for p in self._enc_proj_norm.parameters(): p.requires_grad = True
        self.task_prefix.requires_grad          = True
        self.condition_emb.weight.requires_grad = True
        self._print_trainable("Stage 2")

    def get_stage_2_optimizer(self, enc_lr=5e-5, lora_lr=2e-5, head_lr=1e-5):
        from torch.optim import AdamW
        enc_p, lora_p, head_p = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            if "lora_A" in name or "lora_B" in name: lora_p.append(p)
            elif "lm_head" in name: head_p.append(p)
            else: enc_p.append(p)
        print(f"[Opt S2] enc={len(enc_p)} lr={enc_lr} | "
              f"lora={len(lora_p)} lr={lora_lr} | head={len(head_p)} lr={head_lr}")
        return AdamW([{"params": enc_p,  "lr": enc_lr},
                      {"params": lora_p, "lr": lora_lr},
                      {"params": head_p, "lr": head_lr}], weight_decay=0.05)

    def _print_trainable(self, label):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in self.parameters())
        print(f"[{label}] Trainable: {tr:,} / {tt:,}  ({100*tr/tt:.1f}%)")

    # ── Prefix builder ────────────────────────────────────────
    def _build_prefix(self, eeg, eye, spec, spec_words, condition):
        B = eeg.size(0)
        return torch.cat([
            self.task_prefix.expand(B, -1, -1),
            self.condition_emb(condition).unsqueeze(1),
            self._encode_eeg(eeg, condition),
            self.eye_enc(eye),
            self.spec_enc(spec),
            self.spec_word_enc(spec_words),
        ], dim=1)   # (B, 9, H)

    # ── Forward (teacher-forcing) ─────────────────────────────
    def forward(self, eeg, eye, spec, spec_words, condition, tgt_ids):
        prefix     = self._build_prefix(eeg, eye, spec, spec_words, condition)
        tgt_emb    = self.gpt2.transformer.wte(tgt_ids)
        inputs_emb = torch.cat([prefix, tgt_emb], dim=1)
        logits     = self.gpt2(inputs_embeds=inputs_emb).logits
        return logits[:, prefix.size(1):, :]   # (B, T, V)

    # ── Generation ────────────────────────────────────────────
    @torch.no_grad()
    def generate_text(self, eeg, eye, spec, spec_words, condition, tokenizer,
                      max_len=40, eeg_alpha=0.0, top_k=50,
                      num_beams=1, do_sample=False, top_p=0.9, temperature=1.0):
        self.eval()
        device = eeg.device
        prefix = self._build_prefix(eeg, eye, spec, spec_words, condition)
        B      = prefix.size(0)
        eos    = tokenizer.eos_token_id

        if eeg_alpha > 0:
            eeg_emb = F.normalize(self._encode_eeg(eeg, condition).squeeze(1), dim=-1)
            tok_w   = F.normalize(self.gpt2.transformer.wte.weight.detach(), dim=-1)
            eeg_sim = torch.matmul(eeg_emb, tok_w.T)
        else:
            eeg_sim = None

        eeg_emb_s = F.normalize(self._encode_eeg(eeg, condition).squeeze(1), dim=-1)
        tok_w_s   = F.normalize(self.gpt2.transformer.wte.weight.detach(), dim=-1)
        first_tok = torch.matmul(eeg_emb_s, tok_w_s.T).argmax(dim=-1)

        # ── Beam search ───────────────────────────────────────
        if num_beams > 1:
            prefix_b    = prefix.unsqueeze(1).expand(B, num_beams, -1, -1)
            prefix_b    = prefix_b.contiguous().view(B*num_beams,
                                                     prefix.size(1), prefix.size(2))
            first_tok_b = first_tok.unsqueeze(1).expand(B, num_beams).contiguous().view(B*num_beams)
            generated   = first_tok_b.unsqueeze(1)
            beam_scores        = torch.zeros(B, num_beams, device=device)
            beam_scores[:, 1:] = float("-inf")
            beam_scores        = beam_scores.view(B*num_beams)
            done               = [False]*B
            best_beams         = [None]*B

            for _ in range(max_len):
                tgt_emb    = self.gpt2.transformer.wte(generated)
                inputs_emb = torch.cat([prefix_b, tgt_emb], dim=1)
                logits     = self.gpt2(inputs_embeds=inputs_emb).logits[:, -1, :]

                if eeg_alpha > 0 and eeg_sim is not None:
                    eeg_sim_b = eeg_sim.unsqueeze(1).expand(B, num_beams, -1)\
                                       .contiguous().view(B*num_beams, -1)
                    topk_v, topk_i = torch.topk(logits, top_k, dim=-1)
                    topk_v = topk_v + eeg_alpha * torch.gather(eeg_sim_b, 1, topk_i)
                    adj    = torch.full_like(logits, float("-inf"))
                    logits = adj.scatter_(1, topk_i, topk_v)

                log_probs   = F.log_softmax(logits, dim=-1)
                V           = log_probs.size(-1)
                next_scores = (beam_scores.unsqueeze(1) + log_probs).view(B, num_beams*V)
                top_scores, top_ids = torch.topk(next_scores, num_beams, dim=1)
                beam_idx  = top_ids // V
                token_idx = top_ids  % V

                new_gen = []
                for b in range(B):
                    if done[b]:
                        new_gen.append(generated[b*num_beams: b*num_beams+num_beams])
                        continue
                    beams = []
                    for nb in range(num_beams):
                        bi  = beam_idx[b, nb].item()
                        tid = token_idx[b, nb].item()
                        seq = torch.cat([generated[b*num_beams+bi],
                                         torch.tensor([tid], device=device)])
                        beams.append(seq)
                        if tid == eos and best_beams[b] is None:
                            best_beams[b] = seq
                            done[b]       = True
                    new_gen.append(torch.stack(beams))

                generated   = torch.cat(new_gen, dim=0)
                beam_scores = top_scores.view(B*num_beams)
                if all(done): break

            results = [best_beams[b] if best_beams[b] is not None
                       else generated[b*num_beams] for b in range(B)]
            max_l = max(r.size(0) for r in results)
            pad   = torch.full((B, max_l), eos, dtype=torch.long, device=device)
            for b, r in enumerate(results): pad[b, :r.size(0)] = r
            return pad

        # ── Greedy / Nucleus ──────────────────────────────────
        generated = first_tok.unsqueeze(1)
        for _ in range(max_len):
            tgt_emb    = self.gpt2.transformer.wte(generated)
            inputs_emb = torch.cat([prefix, tgt_emb], dim=1)
            logits     = self.gpt2(inputs_embeds=inputs_emb).logits[:, -1, :]

            if eeg_alpha > 0 and eeg_sim is not None:
                topk_v, topk_i = torch.topk(logits, top_k, dim=-1)
                topk_v = topk_v + eeg_alpha * torch.gather(eeg_sim, 1, topk_i)
                adj    = torch.full_like(logits, float("-inf"))
                logits = adj.scatter_(1, topk_i, topk_v)

            if do_sample:
                logits = logits / max(temperature, 1e-8)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs     = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits    = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                next_tok  = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_tok  = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_tok], dim=1)
            if eos is not None and torch.all(next_tok == eos):
                break

        return generated


# ══════════════════════════════════════════════════════════════════
# MoCo — Momentum Contrast with condition-based hard negatives
# ══════════════════════════════════════════════════════════════════

class MoCoQueue:
    def __init__(self, dim, queue_size=128, device='cpu'):
        self.queue_size = queue_size
        self.device     = device
        self.embeddings = F.normalize(torch.randn(queue_size, dim, device=device), dim=1)
        self.conditions = torch.zeros(queue_size, dtype=torch.long, device=device)
        self.ptr        = 0

    @torch.no_grad()
    def enqueue(self, embeddings, conditions):
        B   = embeddings.size(0)
        end = self.ptr + B
        if end <= self.queue_size:
            self.embeddings[self.ptr:end] = embeddings.detach()
            self.conditions[self.ptr:end] = conditions.detach()
        else:
            first = self.queue_size - self.ptr
            self.embeddings[self.ptr:]            = embeddings[:first].detach()
            self.conditions[self.ptr:]            = conditions[:first].detach()
            self.embeddings[:end-self.queue_size] = embeddings[first:].detach()
            self.conditions[:end-self.queue_size] = conditions[first:].detach()
        self.ptr = end % self.queue_size

    def get_hard_negatives(self, conditions):
        hard_mask = conditions.unsqueeze(1) == self.conditions.unsqueeze(0)
        return self.embeddings, hard_mask


@torch.no_grad()
def _momentum_update(model, momentum_model, m=0.999):
    for p_main, p_mom in zip(model.eeg_enc.parameters(),
                              momentum_model.eeg_enc.parameters()):
        p_mom.data = m * p_mom.data + (1-m) * p_main.data
    for p_main, p_mom in zip(model.contrast_head.parameters(),
                              momentum_model.contrast_head.parameters()):
        p_mom.data = m * p_mom.data + (1-m) * p_main.data


def moco_contrastive_loss(model, momentum_model, queue, batch, device,
                          temperature=0.07, hard_neg_weight=2.0):
    eeg  = batch["eeg"].to(device)
    tgt  = batch["input_ids"].to(device)
    cond = batch["condition"].to(device)
    B    = eeg.size(0)

    eeg_h    = model._encode_eeg(eeg, cond).squeeze(1)
    eeg_proj = F.normalize(model.contrast_head(eeg_h), dim=-1)

    with torch.no_grad():
        _momentum_update(model, momentum_model)
        text_h    = momentum_model.gpt2.transformer.wte(tgt).mean(dim=1)
        text_proj = F.normalize(momentum_model.contrast_head(text_h), dim=-1)

    with torch.no_grad():
        queue_emb, hard_mask = queue.get_hard_negatives(cond)
        queue_emb = queue_emb.clone()

    pos_logits  = (eeg_proj * text_proj).sum(dim=1, keepdim=True) / temperature
    neg_logits  = torch.matmul(eeg_proj, queue_emb.T) / temperature
    weight_mask = torch.where(hard_mask,
                              torch.full_like(neg_logits, hard_neg_weight),
                              torch.ones_like(neg_logits))
    neg_logits  = neg_logits * weight_mask
    logits      = torch.cat([pos_logits, neg_logits], dim=1)
    labels      = torch.zeros(B, dtype=torch.long, device=device)
    loss        = F.cross_entropy(logits, labels)

    with torch.no_grad():
        queue.enqueue(text_proj, cond)
    return loss


# ══════════════════════════════════════════════════════════════════
# Training helpers
# ══════════════════════════════════════════════════════════════════

def run_epoch(model, loader, tokenizer, device,
              optimizer=None, scheduler=None, train=True):
    """SR condition weighting 1.5x + label_smoothing=0.05."""
    from tqdm import tqdm
    model.train() if train else model.eval()
    total = 0
    ctx   = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            eeg   = batch["eeg"].to(device)
            eye   = batch["eye"].to(device)
            spec  = batch["spec"].to(device)
            specw = batch["spec_words"].to(device)
            cond  = batch["condition"].to(device)
            tgt   = batch["input_ids"].to(device)
            logits    = model(eeg, eye, spec, specw, cond, tgt)
            B, T, V   = logits.shape
            base_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, V),
                tgt[:, 1:].contiguous().reshape(-1),
                ignore_index    = tokenizer.pad_token_id,
                label_smoothing = 0.05,
                reduction       = 'none',
            )
            sample_w = torch.ones(B, device=device)
            sample_w[cond == 2] = 1.5
            token_w  = sample_w.unsqueeze(1).expand(B, T-1).contiguous().reshape(-1)
            pad_mask = (tgt[:, 1:].contiguous().reshape(-1) != tokenizer.pad_token_id).float()
            loss     = (base_loss * token_w * pad_mask).sum() / (token_w * pad_mask).sum()
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
            total += loss.item()
    return total / len(loader)


# ══════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════

def _trim(ids, eos, max_len=None):
    ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    if eos in ids: ids = ids[:ids.index(eos)]
    if max_len: ids = ids[:max_len]
    return ids


@torch.no_grad()
def evaluate_bleu_rouge(model, val_loader, tokenizer, device,
                        eeg_alpha=0.0, max_len=40, n_batches=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer as rs
    smoother = SmoothingFunction().method1
    scorer   = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    b1, r1, rl = [], [], []
    model.eval()
    eos = tokenizer.eos_token_id
    for i, batch in enumerate(val_loader):
        if n_batches and i >= n_batches: break
        eeg   = batch["eeg"].to(device)
        eye   = batch["eye"].to(device)
        spec  = batch["spec"].to(device)
        specw = batch["spec_words"].to(device)
        cond  = batch["condition"].to(device)
        refs  = batch["input_ids"]
        gen   = model.generate_text(eeg, eye, spec, specw, cond, tokenizer,
                                    max_len=max_len, eeg_alpha=eeg_alpha)
        for j in range(eeg.size(0)):
            ref_ids = _trim(refs[j], eos)
            gen_ids = _trim(gen[j],  eos, max_len=len(ref_ids))
            r = tokenizer.decode(ref_ids, skip_special_tokens=True)
            h = tokenizer.decode(gen_ids, skip_special_tokens=True)
            rt, ht = r.lower().split(), h.lower().split()
            if not rt or not ht: continue
            b1.append(sentence_bleu([rt], ht, weights=(1,0,0,0),
                                    smoothing_function=smoother))
            rg = scorer.score(r, h)
            r1.append(rg["rouge1"].fmeasure)
            rl.append(rg["rougeL"].fmeasure)
    n = len(b1)
    if n == 0: print("No valid samples."); return None
    out = {"bleu1": sum(b1)/n*100, "rouge1": sum(r1)/n*100, "rougeL": sum(rl)/n*100}
    print(f"Samples : {n}\nBLEU-1  : {out['bleu1']:.2f}%\n"
          f"ROUGE-1 : {out['rouge1']:.2f}%\nROUGE-L : {out['rougeL']:.2f}%")
    return out


@torch.no_grad()
def alpha_sweep(model, val_loader, tokenizer, device,
                alphas=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0], n_batches=20):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1
    eos = tokenizer.eos_token_id
    print(f"{'alpha':>6}  {'FG BLEU-1':>10}\n{'-'*20}")
    model.eval()
    for alpha in alphas:
        scores = []
        for i, batch in enumerate(val_loader):
            if i >= n_batches: break
            eeg   = batch["eeg"].to(device)
            eye   = batch["eye"].to(device)
            spec  = batch["spec"].to(device)
            specw = batch["spec_words"].to(device)
            cond  = batch["condition"].to(device)
            refs  = batch["input_ids"]
            gen   = model.generate_text(eeg, eye, spec, specw, cond, tokenizer,
                                        eeg_alpha=alpha)
            for j in range(eeg.size(0)):
                ref_ids = _trim(refs[j], eos)
                gen_ids = _trim(gen[j],  eos, max_len=len(ref_ids))
                r = tokenizer.decode(ref_ids, skip_special_tokens=True).lower().split()
                h = tokenizer.decode(gen_ids, skip_special_tokens=True).lower().split()
                if r and h:
                    scores.append(sentence_bleu([r], h, weights=(1,0,0,0),
                                                smoothing_function=smoother))
        print(f"{alpha:>6.1f}  "
              f"{sum(scores)/len(scores)*100 if scores else 0.0:>10.2f}%")


# ══════════════════════════════════════════════════════════════════
# Attention Visualization  (updated for HTP tuple return)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_attention(model, batch, tokenizer, device,
                        sample_idx=0, save_prefix="attn"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib; matplotlib.use('Agg')
    except ImportError:
        print("pip install matplotlib"); return

    model.eval()
    eeg   = batch["eeg"][sample_idx:sample_idx+1].to(device)
    eye   = batch["eye"][sample_idx:sample_idx+1].to(device)
    spec  = batch["spec"][sample_idx:sample_idx+1].to(device)
    specw = batch["spec_words"][sample_idx:sample_idx+1].to(device)
    cond  = batch["condition"][sample_idx:sample_idx+1].to(device)
    tgt   = batch["input_ids"][sample_idx:sample_idx+1].to(device)
    cond_name = {0:"NR", 1:"TSR", 2:"SR"}.get(cond[0].item(), "?")

    _ = model(eeg, eye, spec, specw, cond, tgt)
    region_attn_w = model._last_region_attn_w   # list[6] of (local_w, seg_w)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle(f"HTP local attention weights — Condition: {cond_name}", fontsize=13)
    for ax, name, (local_w, seg_w) in zip(axes.flatten(), REGION_NAMES, region_attn_w):
        # local_w: (1, T, 1)
        w_np = local_w[0, :, 0].cpu().numpy()
        ax.plot(w_np, color='steelblue', linewidth=1.2)
        ax.fill_between(range(len(w_np)), w_np, alpha=0.3, color='steelblue')
        ax.set_title(name.replace('_', ' '), fontsize=9)
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("Local attn weight", fontsize=8)
        ax.tick_params(labelsize=7)

        # annotate dominant segment
        sw_np = seg_w[0, :, 0].cpu().numpy()
        dom_seg = sw_np.argmax()
        ax.text(0.97, 0.95, f"dom seg={dom_seg}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="steelblue")

    plt.tight_layout()
    fname = f"{save_prefix}_htp_{cond_name}.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"✅ Saved {fname}")