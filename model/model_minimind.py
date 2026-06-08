import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.pre_alloc_kv = kwargs.get("pre_alloc_kv", False)
        self.use_lookahead = kwargs.get("use_lookahead", False)
        self.lookahead_W = kwargs.get("lookahead_W", 6)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        self.use_pld = kwargs.get("use_pld", False)
        ### TriAttention specific configs
        self.use_tri_attention = kwargs.get("use_tri_attention", False)
        self.tri_threshold = kwargs.get("tri_threshold", 0.05)
        ### StreamingLLM specific configs
        self.use_streaming_llm = kwargs.get("use_streaming_llm", False)
        self.num_sink_tokens = kwargs.get("num_sink_tokens", 4)
        self.sliding_window = kwargs.get("sliding_window", 4096)
        self.use_minference = kwargs.get("use_minference", False)
        self.minference_pattern = kwargs.get("minference_pattern", "auto")
        self.use_mtp = kwargs.get("use_mtp", False)
        self.mtp_depth = kwargs.get("mtp_depth", 3)
        self.mtp_lambda = kwargs.get("mtp_lambda", 0.3)
        ### Medusa specific configs
        self.use_medusa = kwargs.get("use_medusa", False)
        self.medusa_heads = kwargs.get("medusa_heads", 3)
        self.use_kivi = kwargs.get("use_kivi", False)
        self.kivi_bits = kwargs.get("kivi_bits", 2)
        self.kivi_window = kwargs.get("kivi_window", 128)
        self.use_rt_purbo = kwargs.get("use_rt_purbo", False)
        self.use_dflash = kwargs.get("use_dflash", False)
        self.dflash_block_size = kwargs.get("dflash_block_size", 16)
        self.dflash_fusion_layers = kwargs.get("dflash_fusion_layers", [1, 3, 5, 7])
        ### Gated DeltaNet specific configs
        self.use_gated_deltanet = kwargs.get("use_gated_deltanet", False)
        self.gated_deltanet_layers = kwargs.get("gated_deltanet_layers", None)
        ### Qwen3-Next Hybrid specific configs
        self.hybrid_pattern = kwargs.get("hybrid_pattern", None)
        self.partial_rope_dim = kwargs.get("partial_rope_dim", 32)
        ### Lightning Indexer specific configs
        self.use_lightning_indexer = kwargs.get("use_lightning_indexer", False)
        self.lightning_indexer_layers = kwargs.get("lightning_indexer_layers", None)
        self.lightning_indexer_topk = kwargs.get("lightning_indexer_topk", 2048)
        ### NSA (Native Sparse Attention) specific configs
        self.use_nsa = kwargs.get("use_nsa", False)
        self.nsa_block_l = kwargs.get("nsa_block_l", 32)
        self.nsa_sliding_w = kwargs.get("nsa_sliding_w", 512)
        self.nsa_top_n = kwargs.get("nsa_top_n", 16)
        ### mHC specific configs
        self.use_mhc = kwargs.get("use_mhc", False)
        self.mhc_mult = kwargs.get("mhc_mult", 4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        self.capture_pre_rope = False
        self.pre_rope_q = None
        self.pre_rope_k = None
        self.tri_scorer = None
        self.pre_alloc_kv = config.pre_alloc_kv
        self.max_seq_len = config.max_position_embeddings
        self.k_buf = None
        self.v_buf = None
        self.use_streaming_llm = config.use_streaming_llm
        self.num_sink_tokens = config.num_sink_tokens
        self.sliding_window = config.sliding_window
        self.streaming_kv = None
        self.minference_indexer = None
        self.use_kivi = config.use_kivi
        self.kivi_bits = config.kivi_bits
        self.kivi_window = config.kivi_window
        self.kivi_cache = None

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        if self.minference_indexer is not None:
            return self.minference_indexer._minimind_wrapped_forward(
                x, position_embeddings, past_key_value=past_key_value,
                use_cache=use_cache, attention_mask=attention_mask,
            )
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        if self.capture_pre_rope:
            self.pre_rope_q = xq.detach().clone()
            self.pre_rope_k = xk.detach().clone()
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if use_cache and self.use_streaming_llm and bsz == 1:
            if self.streaming_kv is None:
                from .streaming_kv_cache import StreamingKVCache
                self.streaming_kv = StreamingKVCache(
                    num_sink=self.num_sink_tokens,
                    window=self.sliding_window,
                    num_kv_heads=self.n_local_kv_heads,
                    head_dim=self.head_dim,
                    device=xk.device,
                )
            self.streaming_kv.update(xk, xv)
            xk, xv = self.streaming_kv.get_kv()
        elif use_cache and self.use_kivi and bsz == 1:
            if self.kivi_cache is None:
                from .kivi_kv_cache import KIVIKVCache
                self.kivi_cache = KIVIKVCache(
                    num_kv_heads=self.n_local_kv_heads,
                    head_dim=self.head_dim,
                    max_seq_len=self.max_seq_len,
                    window=self.kivi_window,
                    device=xk.device,
                )
            self.kivi_cache.update(xk, xv)
            xk, xv = self.kivi_cache.get()
        elif use_cache and self.pre_alloc_kv and bsz == 1:
            if past_key_value is not None:
                start_pos = past_key_value[0].shape[1]
                self.k_buf[start_pos:start_pos+seq_len] = xk[0]
                self.v_buf[start_pos:start_pos+seq_len] = xv[0]
                cur_len = start_pos + seq_len
                xk = self.k_buf[:cur_len].unsqueeze(0)
                xv = self.v_buf[:cur_len].unsqueeze(0)
            else:
                self.k_buf = xk.new_zeros(self.max_seq_len, self.n_local_kv_heads, self.head_dim)
                self.v_buf = xv.new_zeros(self.max_seq_len, self.n_local_kv_heads, self.head_dim)
                self.k_buf[:seq_len] = xk[0]
                self.v_buf[:seq_len] = xv[0]
        elif past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        use_flash = self.flash and self.tri_scorer is None and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1))
        if use_flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                if self.tri_scorer is not None:
                    L = scores.shape[-1]
                    sparse = self.tri_scorer.make_mask(L).to(scores.device)
                    scores[:, :, :, -L:] += (~sparse).to(scores.dtype) * -1e9
                else:
                    scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    scores += (1.0 - attention_mask) * -1e9
                else:
                    scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        if config.gated_deltanet_layers is not None and layer_id in config.gated_deltanet_layers:
            from .gated_deltanet import GatedDeltaNet
            self.self_attn = GatedDeltaNet(config)
        elif config.lightning_indexer_layers is not None and layer_id in config.lightning_indexer_layers:
            from .lightning_indexer import SparseAttentionWithIndexer
            self.self_attn = SparseAttentionWithIndexer(config)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        if 'position_ids' in kwargs:
            pos_ids = kwargs['position_ids']
            position_embeddings = (self.freqs_cos[pos_ids], self.freqs_sin[pos_ids])
        else:
            position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        if self.config.use_mtp:
            from .mtp_head import MTPHead
            self.mtp_head = MTPHead(self.config)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        if self.config.use_mtp and labels is not None and not kwargs.get('skip_mtp', False):
            mtp_loss = self.mtp_head.compute_loss(hidden_states, labels, input_ids, self.model.embed_tokens)
            loss = loss + self.config.mtp_lambda * mtp_loss
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, use_pld=False, use_mtp=False, **kwargs):
        pre_alloc_kv = kwargs.pop("pre_alloc_kv", self.config.pre_alloc_kv)
        if pre_alloc_kv:
            for block in self.model.layers:
                block.self_attn.pre_alloc_kv = True
                block.self_attn.k_buf = None
                block.self_attn.v_buf = None
        if self.config.use_streaming_llm:
            for block in self.model.layers:
                block.self_attn.streaming_kv = None
        if self.config.use_kivi:
            for block in self.model.layers:
                block.self_attn.kivi_cache = None
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())

        use_lookahead = self.config.use_lookahead and use_cache and num_return_sequences == 1
        lookahead = None
        if use_lookahead:
            from model.lookahead_decoding import LookaheadDecoding
            lookahead = LookaheadDecoding(self, None, W=self.config.lookahead_W)

        use_pld = use_pld and use_cache and num_return_sequences == 1
        pld_decoder = None
        if use_pld:
            from .pld_decoding import PLDDecoder
            prompt_ids = input_ids[0].clone()
            pld_decoder = PLDDecoder(max_ngram=4, num_pred=8)

        use_mtp = use_mtp and use_cache and num_return_sequences == 1 and hasattr(self, 'mtp_head')

        medusa_heads = kwargs.pop("medusa_heads", None)
        use_medusa = medusa_heads is not None and use_cache and num_return_sequences == 1
        medusa_decoder = None
        if use_medusa:
            from .medusa_heads import MedusaDecoder
            medusa_decoder = MedusaDecoder(medusa_heads, tree_topk=1)

        def _truncate_kv(past_key_values, target_len):
            if past_key_values is None:
                return None
            new = []
            for layer_kv in past_key_values:
                if layer_kv is not None:
                    k, v = layer_kv
                    new.append((k[:, :target_len].contiguous(), v[:, :target_len].contiguous()))
                else:
                    new.append(None)
            return new

        def _ar_step():
            nonlocal input_ids, attention_mask, past_key_values
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            if use_medusa:
                medusa_decoder.update_hidden(outputs.hidden_states)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            return next_token

        generated = 0
        while generated < max_new_tokens:
            if use_pld and pld_decoder is not None and not finished.any():
                result = pld_decoder.step(self, input_ids, past_key_values, prompt_ids, attention_mask)
                if result is not None:
                    new_tokens_list, new_past_kv = result
                    num_new = len(new_tokens_list)
                    if generated + num_new > max_new_tokens:
                        num_new = max_new_tokens - generated
                        new_tokens_list = new_tokens_list[:num_new]
                    generated += num_new
                    new_tokens = torch.tensor([new_tokens_list], device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                    past_key_values = new_past_kv
                    if attention_mask is not None:
                        extra = attention_mask.new_ones(attention_mask.shape[0], num_new)
                        attention_mask = torch.cat([attention_mask, extra], dim=-1)
                    for t in new_tokens_list:
                        if streamer: streamer.put(torch.tensor([[t]], device='cpu'))
                    if eos_token_id is not None:
                        for t in new_tokens_list:
                            if t == eos_token_id:
                                finished = torch.ones_like(finished)
                                break
                        if finished.all():
                            break
                    continue

            if use_lookahead:
                new_tokens, new_pkv = lookahead.step(
                    input_ids, past_key_values, attention_mask,
                    temperature, top_p, top_k, do_sample
                )
                if new_tokens is not None:
                    num_new = new_tokens.shape[1]
                    if generated + num_new > max_new_tokens:
                        num_new = max_new_tokens - generated
                        new_tokens = new_tokens[:, :num_new]
                    generated += num_new
                    input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                    past_key_values = new_pkv
                    if attention_mask is not None:
                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], num_new)], -1)
                    for i in range(num_new):
                        if streamer: streamer.put(new_tokens[:, i:i+1].cpu())
                    if eos_token_id is not None:
                        for b in range(input_ids.shape[0]):
                            if finished[b]: continue
                            if (new_tokens[b] == eos_token_id).any():
                                finished[b] = True
                        if finished.all(): break
                    continue

            if use_medusa and not finished.any():
                new_tokens, new_pkv = medusa_decoder.step(
                    self, input_ids, past_key_values,
                    temperature, top_p, top_k, do_sample,
                )
                if new_tokens is not None:
                    num_new = new_tokens.shape[1]
                    if generated + num_new > max_new_tokens:
                        num_new = max_new_tokens - generated
                        new_tokens = new_tokens[:, :num_new]
                    generated += num_new
                    input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                    past_key_values = new_pkv
                    if attention_mask is not None:
                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], num_new)], -1)
                    for i in range(num_new):
                        if streamer: streamer.put(new_tokens[:, i:i+1].cpu())
                    if eos_token_id is not None:
                        for b in range(input_ids.shape[0]):
                            if finished[b]: continue
                            if (new_tokens[b] == eos_token_id).any():
                                finished[b] = True
                        if finished.all(): break
                    continue

            if use_mtp and not finished.any():
                past_len = past_key_values[0][0].shape[1] if past_key_values else 0
                mtp_outputs = self.forward(
                    input_ids[:, past_len:], attention_mask, past_key_values,
                    use_cache=use_cache, skip_mtp=True, **kwargs
                )
                mtp_attn = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
                h_last = mtp_outputs.hidden_states[:, -1:]
                mtp_kv = mtp_outputs.past_key_values
                D = self.config.mtp_depth
                draft_tokens = []
                h_cur = h_last.squeeze(1)
                for _ in range(D):
                    l = self.mtp_head.forward(h_cur)
                    t = l.argmax(dim=-1, keepdim=True)
                    draft_tokens.append(t.item())
                    h_cur = h_last.squeeze(1) + self.model.embed_tokens(t).squeeze(1)
                if len(draft_tokens) > 0:
                    draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
                    verif_outputs = self.forward(
                        draft_tensor, attention_mask=None,
                        past_key_values=mtp_kv, use_cache=True, skip_mtp=True
                    )
                    verif_logits = verif_outputs.logits[0]
                    verif_kv = verif_outputs.past_key_values
                    preds = verif_logits.argmax(dim=-1)
                    accept_len = 0
                    for i in range(D - 1):
                        if preds[i].item() == draft_tokens[i + 1]:
                            accept_len += 1
                        else:
                            break
                    final_tokens = draft_tokens[:accept_len + 1] + [preds[accept_len].item()]
                    num_new = len(final_tokens)
                    if generated + num_new > max_new_tokens:
                        num_new = max_new_tokens - generated
                        final_tokens = final_tokens[:num_new]
                    generated += num_new
                    new_tensor = torch.tensor([final_tokens], device=input_ids.device)
                    input_ids = torch.cat([input_ids, new_tensor], dim=-1)
                    full_len = input_ids.shape[1] - 1
                    past_key_values = _truncate_kv(verif_kv, full_len)
                    if attention_mask is not None:
                        extra = attention_mask.new_ones(attention_mask.shape[0], num_new)
                        attention_mask = torch.cat([attention_mask, extra], dim=-1)
                    for t in final_tokens:
                        if streamer: streamer.put(torch.tensor([[t]], device='cpu'))
                    if eos_token_id is not None:
                        for t in final_tokens:
                            if t == eos_token_id:
                                finished = torch.ones_like(finished)
                                break
                        if finished.all():
                            break
                    continue

            next_token = _ar_step()
            generated += 1
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids