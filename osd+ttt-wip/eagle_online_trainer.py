import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from typing import Dict, List
import logging
import sys
import os

# Import SGLang specific layers for alignment
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.rotary_embedding import get_rope, apply_rotary_pos_emb
from sglang.srt.speculative.online_structs import collate_ttt_experiences

# Use the server args init function
from sglang.srt.server_args import set_global_server_args_for_scheduler

logger = logging.getLogger(__name__)

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
        
        # RoPE Init using SGLang util
        self.rotary_dim = int(getattr(config, "partial_rotary_factor", 1) * self.head_dim)
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_theta = getattr(config, "rope_theta", 10000)
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.rope_is_neox_style = getattr(config, "rope_is_neox_style", True)

        # Initialize RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=self.rope_is_neox_style,
            rope_scaling=self.rope_scaling,
            dtype=torch.float32, 
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_bias = getattr(config, "attention_bias", False) or getattr(config, "qkv_bias", False)

        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(
            2 * config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=self.use_bias
        )
        self.self_attn.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=self.use_bias)

        if getattr(config, "model_type", "") == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size
            
        self.mlp = nn.Module()
        mlp_bias = getattr(config, "mlp_bias", False)
        self.mlp.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter_size, bias=mlp_bias)
        self.mlp.down_proj = nn.Linear(inter_size, config.hidden_size, bias=mlp_bias)
        
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states, embeds, position_ids):
        # position_ids: [Batch] (absolute position)
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
        # Reshape to [B, SeqLen=1, NumHeads, HeadDim]
        q = q.view(B, 1, self.num_heads, self.head_dim)
        k = k.view(B, 1, self.num_kv_heads, self.head_dim)
        v = v.view(B, 1, self.num_kv_heads, self.head_dim)

        # Ensure cache device match
        if self.rotary_emb.cos_sin_cache.device != q.device:
             self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(q.device)

        # Apply SGLang RoPE
        # input positions: [B] (SGLang native forward handles this by expanding)
        q, k = self.rotary_emb(position_ids, q, k)

        # Transpose for SDP: [B, NumHeads, SeqLen=1, HeadDim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA expansion
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # [B, H, 1, D] -> [B, H*D]
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

class EagleShadowModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # EAGLE3 Fusion: target_hidden_size * 3 -> hidden_size
        if hasattr(config, "target_hidden_size"):
            hidden_size_in = config.target_hidden_size
        else:
            hidden_size_in = config.hidden_size

        self.fc = nn.Linear(
            hidden_size_in * 3, 
            config.hidden_size, 
            bias=getattr(config, "bias", False)
        )
        # Corresponds to model.layers.0 in SGLang structure (but only 1 layer for draft)
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward_step(self, hidden_states, token_embeds, position_ids):
        """
        One step of autoregressive generation.
        """
        hidden_states = self.midlayer(hidden_states, token_embeds, position_ids)
        hidden_states_normed = self.norm(hidden_states)
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
        # Configure Logging for Shadow Process
        logging.basicConfig(
            level=logging.INFO,
            format=f"[ShadowTrainer-PID{os.getpid()}] %(asctime)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
            force=True
        )

        try:
            # Init SGLang Global Args (Important for RoPE)
            set_global_server_args_for_scheduler(self.server_args)

            if torch.cuda.is_available():
                torch.cuda.set_device(self.device)

            # 1. Build Model
            model = EagleShadowModel(self.model_config.hf_config).to(self.device)
            
            # 2. Load Weights (exclude lm_head which we take from target)
            filtered_weights = {
                k: v for k, v in self.draft_weights.items() 
                if "embed_tokens" not in k and "lm_head" not in k
            }
            # Remap keys if necessary. 
            # In eagle_utils.py we map Shadow -> Inference. Here we load Inference -> Shadow.
            # Shadow: model.midlayer... 
            # Inference: model.layers.0...
            # We need to reverse map the keys from draft_weights to match ShadowModel
            remapped_load = {}
            for k, v in filtered_weights.items():
                new_k = k.replace("model.layers.0.", "model.midlayer.")
                remapped_load[new_k] = v
            
            missing, unexpected = model.load_state_dict(remapped_load, strict=False)
            if missing:
                logger.info(f"Missing keys (expected for embeddings/head): {missing}")

            # 3. Setup Frozen LM Head from Target
            if model.lm_head.weight.shape != self.target_head_weights.shape:
                vocab, dim = self.target_head_weights.shape
                model.lm_head = nn.Linear(dim, vocab, bias=False).to(self.device)
            
            model.lm_head.weight.data.copy_(self.target_head_weights.to(self.device))
            model.lm_head.requires_grad_(False)

            # 4. Optimizer
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr
            )

            logger.info(f"Ready on {self.device}")

            while True:
                cmd, payload = self.data_queue.get()

                if cmd == "STOP":
                    break

                if cmd == "TRAIN_AND_SYNC":
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

                    # Unpack
                    teacher_feat_g = batch["teacher_feat_g"] 
                    start_token_embed = batch["start_token_embed"] 
                    target_logits = batch["target_logits_at_end"]
                    position_ids = batch["start_position_ids"] 
                    padded_embeds = batch["padded_embeds"] 
                    lengths = batch["lengths"]
                    
                    # === TTT Rollout ===
                    
                    # Step 1: Projection + Decode
                    current_state = model.fc(teacher_feat_g)
                    
                    logits, current_state = model.forward_step(
                        current_state, 
                        start_token_embed, 
                        position_ids
                    )
                    
                    # Initialize final_logits with Step 1 output
                    # This handles the case where lengths[i] == 0 (rejection at step 1)
                    final_logits = logits.clone()
                    
                    max_len = padded_embeds.shape[1]
                    
                    for t in range(max_len):
                        # Get next input embeddings for all batch items
                        next_embed = padded_embeds[:, t, :]
                        
                        # Increment position ID
                        position_ids = position_ids + 1
                        
                        step_logits, current_state = model.forward_step(
                            current_state,
                            next_embed,
                            position_ids
                        )
                        
                        # Update final_logits for sequences that END at this step.
                        # t is 0-based index of the loop.
                        # If lengths[b] == 1, it means we accepted 1 token (t+1).
                        # So we run loop t=0 once. The input was accepted_token[0].
                        # The output logit predicts t+2.
                        # So if lengths[b] == t + 1, we save this logit.
                        mask = (lengths == (t + 1))
                        if mask.any():
                            final_logits[mask] = step_logits[mask]

                    # Loss Calculation
                    T = self.loss_temp
                    loss = F.kl_div(
                        F.log_softmax(final_logits / T, dim=-1),
                        F.softmax(target_logits / T, dim=-1),
                        reduction="batchmean"
                    ) * (T * T)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Return updated weights (only trainable ones)
                    ret_weights = {
                        k: v.cpu() for k, v in model.state_dict().items() 
                        if "lm_head" not in k and "embed_tokens" not in k
                    }
                    self.result_queue.put((ret_weights, loss.item()))

        except Exception as e:
            import traceback
            logger.error(f"ShadowTrainer CRASH: {e}\n{traceback.format_exc()}")
            self.result_queue.put(({}, None))
