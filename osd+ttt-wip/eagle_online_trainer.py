# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.multiprocessing as mp
# from typing import Dict, List
# import logging
# import sys
# import os

# from sglang.srt.layers.activation import SiluAndMul
# from sglang.srt.layers.rotary_embedding import get_rope
# from sglang.srt.speculative.online_structs import collate_ttt_experiences
# from sglang.srt.server_args import set_global_server_args_for_scheduler

# logger = logging.getLogger(__name__)

# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)


# class Eagle3DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
#         self.num_kv_heads = config.num_key_value_heads
#         self.num_kv_groups = self.num_heads // self.num_kv_heads

#         # RoPE Init
#         self.rotary_dim = int(getattr(config, "partial_rotary_factor", 1) * self.head_dim)
#         self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
#         self.rope_theta = getattr(config, "rope_theta", 10000)
#         self.rope_scaling = getattr(config, "rope_scaling", None)
#         self.rope_is_neox_style = getattr(config, "rope_is_neox_style", True)

#         self.rotary_emb = get_rope(
#             head_size=self.head_dim,
#             rotary_dim=self.rotary_dim,
#             max_position=self.max_position_embeddings,
#             base=self.rope_theta,
#             is_neox_style=self.rope_is_neox_style,
#             rope_scaling=self.rope_scaling,
#             dtype=torch.float32,
#         )

#         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#         self.use_bias = getattr(config, "attention_bias", False) or getattr(config, "qkv_bias", False)

#         # Self Attention
#         self.self_attn = nn.Module()
#         self.self_attn.qkv_proj = nn.Linear(
#             2 * config.hidden_size,
#             (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
#             bias=self.use_bias
#         )
#         self.self_attn.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=self.use_bias)

#         # MLP
#         if getattr(config, "model_type", "") == "llama4_text":
#             inter_size = config.intermediate_size_mlp
#         else:
#             inter_size = config.intermediate_size

#         self.mlp = nn.Module()
#         mlp_bias = getattr(config, "mlp_bias", False)
#         # Use merged GateUp to align with SGLang
#         self.mlp.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter_size, bias=mlp_bias)
#         self.mlp.down_proj = nn.Linear(inter_size, config.hidden_size, bias=mlp_bias)

#         self.act_fn = SiluAndMul()

#     def forward(self, hidden_states, embeds, position_ids):
#         # inputs: [Batch, Hidden]
#         residual = hidden_states
#         normed_embeds = self.input_layernorm(embeds)
#         normed_hidden = self.hidden_norm(hidden_states)
#         attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)

#         qkv = self.self_attn.qkv_proj(attn_input)
#         q, k, v = torch.split(
#             qkv,
#             [
#                 self.num_heads * self.head_dim,
#                 self.num_kv_heads * self.head_dim,
#                 self.num_kv_heads * self.head_dim,
#             ],
#             dim=-1
#         )

#         B = q.shape[0]
#         q = q.view(B, 1, self.num_heads, self.head_dim)
#         k = k.view(B, 1, self.num_kv_heads, self.head_dim)
#         v = v.view(B, 1, self.num_kv_heads, self.head_dim)

#         if hasattr(self.rotary_emb, "cos_sin_cache") and self.rotary_emb.cos_sin_cache.device != q.device:
#             self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(q.device)

#         q, k = self.rotary_emb(position_ids, q, k)

#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         if self.num_kv_groups > 1:
#             k = k.repeat_interleave(self.num_kv_groups, dim=1)
#             v = v.repeat_interleave(self.num_kv_groups, dim=1)

#         attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

#         attn_out = attn_out.transpose(1, 2).reshape(B, self.num_heads * self.head_dim)
#         attn_out = self.self_attn.o_proj(attn_out)

#         hidden_states = residual + attn_out
#         residual = hidden_states

#         normed_hidden = self.post_attention_layernorm(hidden_states)
#         gate_up = self.mlp.gate_up_proj(normed_hidden)
#         mlp_out = self.act_fn.forward_native(gate_up)
#         mlp_out = self.mlp.down_proj(mlp_out)

#         hidden_states = residual + mlp_out
#         return hidden_states


# class EagleInnerModel(nn.Module):
#     """
#     Wrapper to align with SGLang model.model structure
#     """

#     def __init__(self, config):
#         super().__init__()
#         if hasattr(config, "target_hidden_size"):
#             hidden_size_in = config.target_hidden_size
#         else:
#             hidden_size_in = config.hidden_size

#         self.fc = nn.Linear(
#             hidden_size_in * 3,
#             config.hidden_size,
#             bias=getattr(config, "bias", False)
#         )
#         self.midlayer = Eagle3DecoderLayer(config)
#         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


# class EagleShadowModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.model = EagleInnerModel(config)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#     def forward_step(self, hidden_states, token_embeds, position_ids):
#         hidden_states = self.model.midlayer(hidden_states, token_embeds, position_ids)
#         hidden_states_normed = self.model.norm(hidden_states)
#         logits = self.lm_head(hidden_states_normed)
#         return logits, hidden_states


# class ShadowTrainerProcess(mp.Process):
#     def __init__(
#             self,
#             server_args,
#             model_config,
#             draft_weights: Dict[str, torch.Tensor],
#             target_head_weights: torch.Tensor,
#             data_queue: mp.Queue,
#             result_queue: mp.Queue,
#             device="cuda:0",
#     ):
#         super().__init__()
#         self.server_args = server_args
#         self.model_config = model_config
#         self.draft_weights = draft_weights
#         self.target_head_weights = target_head_weights
#         self.data_queue = data_queue
#         self.result_queue = result_queue
#         self.device = device
#         self.lr = server_args.online_ttt_lr
#         self.loss_temp = 2.0

#     def run(self):
#         # Configure logging
#         logging.basicConfig(
#             level=logging.INFO,
#             format=f"[ShadowTrainer-PID{os.getpid()}] %(asctime)s - %(levelname)s - %(message)s",
#             stream=sys.stdout,
#             force=True
#         )

#         try:
#             set_global_server_args_for_scheduler(self.server_args)

#             if torch.cuda.is_available():
#                 torch.cuda.set_device(self.device)

#             model = EagleShadowModel(self.model_config.hf_config).to(self.device)

#             # Load weights
#             filtered_weights = {
#                 k: v for k, v in self.draft_weights.items()
#                 if "embed_tokens" not in k and "lm_head" not in k
#             }

#             remapped_load = {}
#             for k, v in filtered_weights.items():
#                 # Map SGLang "model.layers.0" -> Shadow "model.midlayer"
#                 new_k = k.replace("model.layers.0.", "model.midlayer.")
#                 remapped_load[new_k] = v

#             missing, unexpected = model.load_state_dict(remapped_load, strict=False)
#             if missing:
#                 logger.info(f"Missing keys: {missing}")
#             if unexpected:
#                 logger.info(f"Unexpected keys: {unexpected}")

#             # Freeze LM Head
#             if model.lm_head.weight.shape != self.target_head_weights.shape:
#                 vocab, dim = self.target_head_weights.shape
#                 model.lm_head = nn.Linear(dim, vocab, bias=False).to(self.device)

#             model.lm_head.weight.data.copy_(self.target_head_weights.to(self.device))
#             model.lm_head.requires_grad_(False)

#             # Optimizer: Train FC + MidLayer + Norm
#             trainable_params = []
#             for n, p in model.named_parameters():
#                 if "lm_head" in n:
#                     p.requires_grad = False
#                 else:
#                     p.requires_grad = True
#                     trainable_params.append(p)

#             optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

#             logger.info(f"Ready on {self.device}. Trainable params: {len(trainable_params)}")

#             while True:
#                 cmd, payload = self.data_queue.get()

#                 if cmd == "STOP":
#                     break

#                 if cmd == "TRAIN_AND_SYNC":
#                     experiences = payload
#                     if not experiences:
#                         self.result_queue.put(({}, None))
#                         continue

#                     batch = collate_ttt_experiences(experiences, self.device)
#                     if not batch:
#                         self.result_queue.put(({}, None))
#                         continue

#                     model.train()
#                     optimizer.zero_grad()

#                     teacher_feat_g = batch["teacher_feat_g"]
#                     start_token_embed = batch["start_token_embed"]
#                     target_logits = batch["target_logits_at_end"]
#                     position_ids = batch["start_position_ids"]
#                     padded_embeds = batch["padded_embeds"]
#                     lengths = batch["lengths"]

#                     # Step 1
#                     current_state = model.model.fc(teacher_feat_g)

#                     logits, current_state = model.forward_step(
#                         current_state,
#                         start_token_embed,
#                         position_ids
#                     )

#                     final_logits = logits.clone()
#                     max_len = padded_embeds.shape[1]

#                     for t in range(max_len):
#                         next_embed = padded_embeds[:, t, :]
#                         position_ids = position_ids + 1

#                         step_logits, current_state = model.forward_step(
#                             current_state,
#                             next_embed,
#                             position_ids
#                         )

#                         mask = (lengths == (t + 1))
#                         if mask.any():
#                             final_logits[mask] = step_logits[mask]

#                     T = self.loss_temp
#                     loss = F.kl_div(
#                         F.log_softmax(final_logits / T, dim=-1),
#                         F.softmax(target_logits / T, dim=-1),
#                         reduction="batchmean"
#                     ) * (T * T)

#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                     optimizer.step()

#                     # Return updated weights (FC + MidLayer + Norm)
#                     ret_weights = {}
#                     for k, v in model.state_dict().items():
#                         if "lm_head" not in k and "embed_tokens" not in k:
#                             # Map back: Shadow "model.midlayer" -> SGLang "model.layers.0"
#                             new_k = k.replace("model.midlayer.", "model.layers.0.")
#                             ret_weights[new_k] = v.cpu()

#                     self.result_queue.put((ret_weights, loss.item()))

#         except Exception as e:
#             import traceback
#             logger.error(f"ShadowTrainer CRASH: {e}\n{traceback.format_exc()}")
#             self.result_queue.put(({}, None))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import Dict, List
import logging
import sys
import os
from torch.cuda.amp import autocast
from sglang.srt.layers.activation import SiluAndMul
# [REMOVED] Removed SGLang RoPE dependency to avoid vllm requirement and ensure Autograd support
# from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.speculative.online_structs import collate_ttt_experiences
from sglang.srt.server_args import set_global_server_args_for_scheduler

logger = logging.getLogger(__name__)

# --- [Added] Pure PyTorch RoPE for Training ---
class TrainingRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim, theta=10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        # Precompute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions, q, k):
        # positions: [B]
        # q, k: [B, 1, H, D] (Sequence length is 1 during TTT step)
        
        # Compute cos/sin for the given positions
        # inv_freq: [D/2]
        inv_freq = self.inv_freq.to(positions.device)
        
        # Outer product: [B, D/2]
        sinusoid_inp = torch.outer(positions.float(), inv_freq)
        
        # [B, D/2]
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        # Expand to [B, D] to match standard Llama RoPE concatenation style
        # cat([sin, sin]) matches the behavior of 'rotate_half' which pairs x[i] with x[i+N/2]
        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
        
        # Reshape to broadcast over heads: [B, 1, 1, D]
        # Note: q.shape[0] is B
        sin = sin.view(q.shape[0], 1, 1, self.rotary_dim).type_as(q)
        cos = cos.view(q.shape[0], 1, 1, self.rotary_dim).type_as(q)
        
        # Helper for rotation
        def rotate_half(x):
            x1 = x[..., :x.shape[-1]//2]
            x2 = x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        # Apply RoPE to the rotary part of q and k
        # Support partial rotary embedding (rotary_dim <= head_dim)
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
        
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        
        return q, k
# ----------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # RoPE Init (Using Pure PyTorch for Training)
        self.rotary_dim = int(getattr(config, "partial_rotary_factor", 1) * self.head_dim)
        self.rope_theta = getattr(config, "rope_theta", 10000)
        
        # Replaced get_rope with TrainingRotaryEmbedding
        self.rotary_emb = TrainingRotaryEmbedding(
            rotary_dim=self.rotary_dim,
            theta=self.rope_theta
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_bias = getattr(config, "attention_bias", False) or getattr(config, "qkv_bias", False)

        # Self Attention
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(
            2 * config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=self.use_bias
        )
        self.self_attn.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=self.use_bias)

        # MLP
        if getattr(config, "model_type", "") == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size
            
        self.mlp = nn.Module()
        mlp_bias = getattr(config, "mlp_bias", False)
        # Use merged GateUp to align with SGLang
        self.mlp.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter_size, bias=mlp_bias)
        self.mlp.down_proj = nn.Linear(inter_size, config.hidden_size, bias=mlp_bias)
        
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states, embeds, position_ids):
        # inputs: [Batch, Hidden]
        residual = hidden_states
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)
        attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)

        qkv = self.self_attn.qkv_proj(attn_input)
        q, k, v = torch.split(
            qkv,
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1
        )

        B = q.shape[0]
        q = q.view(B, 1, self.num_heads, self.head_dim)
        k = k.view(B, 1, self.num_kv_heads, self.head_dim)
        v = v.view(B, 1, self.num_kv_heads, self.head_dim)

        # Call Training RoPE
        q, k = self.rotary_emb(position_ids, q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, self.num_heads * self.head_dim)
        attn_out = self.self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out
        residual = hidden_states

        normed_hidden = self.post_attention_layernorm(hidden_states)
        gate_up = self.mlp.gate_up_proj(normed_hidden)
        mlp_out = self.act_fn.forward_native(gate_up)
        mlp_out = self.mlp.down_proj(mlp_out)

        hidden_states = residual + mlp_out
        return hidden_states

class EagleInnerModel(nn.Module):
    """
    Wrapper to align with SGLang model.model structure
    """
    def __init__(self, config):
        super().__init__()
        if hasattr(config, "target_hidden_size"):
            hidden_size_in = config.target_hidden_size
        else:
            hidden_size_in = config.hidden_size

        self.fc = nn.Linear(
            hidden_size_in * 3, 
            config.hidden_size, 
            bias=getattr(config, "bias", False)
        )
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class EagleShadowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = EagleInnerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward_step(self, hidden_states, token_embeds, position_ids):
        hidden_states = self.model.midlayer(hidden_states, token_embeds, position_ids)
        hidden_states_normed = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states_normed)
        return logits, hidden_states

class ShadowTrainerProcess(mp.Process):
    def __init__(
            self,
            server_args,
            model_config,
            draft_weights: Dict[str, torch.Tensor],
            target_head_weights: torch.Tensor,
            data_queue: mp.Queue,
            result_queue: mp.Queue,
            device="cuda:0",
    ):
        super().__init__()
        self.server_args = server_args
        self.model_config = model_config
        self.draft_weights = draft_weights
        self.target_head_weights = target_head_weights
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.device = device
        self.lr = server_args.online_ttt_lr
        self.loss_temp = 2.0 

    def run(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=f"[ShadowTrainer-PID{os.getpid()}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
            force=True
        )
    
        try:
            set_global_server_args_for_scheduler(self.server_args)
    
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device)
    
            model = EagleShadowModel(self.model_config.hf_config).to(self.device)
    
            # -------------------- Load weights (from draft inference) --------------------
            filtered_weights = {
                k: v for k, v in self.draft_weights.items()
                if "embed_tokens" not in k and "lm_head" not in k
            }
    
            remapped_load = {}
            for k, v in filtered_weights.items():
                # SGLang: "model.layers.0." -> Shadow: "model.midlayer."
                new_k = k.replace("model.layers.0.", "model.midlayer.")
                remapped_load[new_k] = v
    
            missing, unexpected = model.load_state_dict(remapped_load, strict=False)
            if missing:
                logger.info(f"Missing keys: {missing}")
            if unexpected:
                logger.info(f"Unexpected keys: {unexpected}")
    
            # -------------------- LM Head freeze & resize --------------------
            if model.lm_head.weight.shape != self.target_head_weights.shape:
                vocab, dim = self.target_head_weights.shape
                model.lm_head = nn.Linear(dim, vocab, bias=False).to(self.device)
    
            model.lm_head.weight.data.copy_(self.target_head_weights.to(self.device))
            model.lm_head.requires_grad_(False)
    
            # -------------------- Optimizer: train FC + MidLayer + Norm --------------------
            trainable_params = []
            for n, p in model.named_parameters():
                if "lm_head" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
                    trainable_params.append(p)
    
            optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
            logger.info(f"Ready on {self.device}. Trainable params: {len(trainable_params)}")
    
            # For once-only logs
            logged_fc_path = False
            logged_bypass_path = False
    
            # Param dtype for inputs alignment
            param_dtype = next(p for p in model.parameters() if p.requires_grad).dtype
    
            # Cached dims
            hidden_size = model.model.midlayer.hidden_size
            fc_in_features = model.model.fc.in_features  # expected 3 * hidden_size
    
            while True:
                cmd, payload = self.data_queue.get()
                if cmd == "STOP":
                    break
    
                if cmd != "TRAIN_AND_SYNC":
                    continue
    
                experiences = payload
                if not experiences:
                    self.result_queue.put(({}, None))
                    continue
    
                batch = collate_ttt_experiences(experiences, self.device)
                if not batch:
                    self.result_queue.put(({}, None))
                    continue
    
                model.train()
                optimizer.zero_grad()
    
                # -------------------- Fetch batch tensors --------------------
                teacher_feat_g = batch["teacher_feat_g"]           # [B, D]  (D=3*H or H)
                start_token_embed = batch["start_token_embed"]     # [B, H]
                target_logits = batch["target_logits_at_end"]      # [B, V]
                position_ids = batch["start_position_ids"]         # [B] long
                padded_embeds = batch["padded_embeds"]             # [B, T, H]
                lengths = batch["lengths"]                         # [B] int
    
                # -------------------- dtype/device alignment --------------------
                teacher_feat_g = teacher_feat_g.to(self.device, dtype=param_dtype)
                start_token_embed = start_token_embed.to(self.device, dtype=param_dtype)
                padded_embeds = padded_embeds.to(self.device, dtype=param_dtype)
    
                # position_ids 必须是 long
                position_ids = position_ids.to(self.device, dtype=torch.long)
    
                # KL 在 float32 计算更稳定
                target_logits_f32 = target_logits.to(self.device, dtype=torch.float32)
    
                # -------------------- Step 0: choose init state path --------------------
                # If teacher_feat_g = concat([l, m, h]) with shape 3*H -> go through fc
                # Else if already fused g with shape H -> bypass fc
                if teacher_feat_g.shape[-1] == fc_in_features:
                    if not logged_fc_path:
                        logger.info(f"[TTT] Using FC fuse path: teacher_feat_g dim={teacher_feat_g.shape[-1]} -> fc({fc_in_features}->{hidden_size})")
                        logged_fc_path = True
                    current_state = model.model.fc(teacher_feat_g)  # [B, H]
                elif teacher_feat_g.shape[-1] == hidden_size:
                    if not logged_bypass_path:
                        logger.info(f"[TTT] Using BYPASS path: teacher_feat_g already fused to hidden_size={hidden_size}")
                        logged_bypass_path = True
                    current_state = teacher_feat_g  # already [B, H]
                else:
                    # 极端兜底：线性适配到 H（只创建一次并缓存）
                    if not hasattr(model, "_ttt_adapt_to_hidden"):
                        in_dim = teacher_feat_g.shape[-1]
                        logger.warning(f"[TTT] Unexpected teacher_feat_g dim {in_dim}, creating adapter to {hidden_size}.")
                        adapter = nn.Linear(in_dim, hidden_size, bias=False).to(self.device).to(param_dtype)
                        # 小初始化：近似恒等（若 in_dim != H，则随机小权重）
                        nn.init.xavier_uniform_(adapter.weight, gain=1.0)
                        model._ttt_adapt_to_hidden = adapter
                    current_state = model._ttt_adapt_to_hidden(teacher_feat_g)
    
                # -------------------- Unroll along accepted path --------------------
                logits, current_state = model.forward_step(
                    current_state,
                    start_token_embed,
                    position_ids
                )
    
                # 只在最终时刻计算 KL（“first reject”对齐）
                final_logits = logits.clone()
    
                max_len = padded_embeds.shape[1]
                for t in range(max_len):
                    if max_len == 0:
                        break
                    next_embed = padded_embeds[:, t, :]
                    position_ids = position_ids + 1  # long
    
                    step_logits, current_state = model.forward_step(
                        current_state,
                        next_embed,
                        position_ids
                    )
    
                    # lengths == t+1 的样本在该步结束（拿到最后一步 logits）
                    mask = (lengths == (t + 1))
                    if mask.any():
                        final_logits[mask] = step_logits[mask]
    
                # -------------------- KL loss (float32) --------------------
                T = self.loss_temp
                final_logits_f32 = final_logits.to(torch.float32)
    
                loss = F.kl_div(
                    F.log_softmax(final_logits_f32 / T, dim=-1),
                    F.softmax(target_logits_f32 / T, dim=-1),
                    reduction="batchmean"
                ) * (T * T)
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
    
                # -------------------- Return updated weights (FC + MidLayer + Norm) --------------------
                ret_weights = {}
                for k, v in model.state_dict().items():
                    if "lm_head" in k or "embed_tokens" in k:
                        continue
                    # Shadow "model.midlayer." -> SGLang "model.layers.0."
                    new_k = k.replace("model.midlayer.", "model.layers.0.")
                    ret_weights[new_k] = v.detach().to("cpu")
    
                self.result_queue.put((ret_weights, float(loss.item())))
    
        except Exception as e:
            import traceback
            logger.error(f"ShadowTrainer CRASH: {e}\n{traceback.format_exc()}")
            self.result_queue.put(({}, None))
