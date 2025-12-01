# python/sglang/srt/speculative/shadow_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Dict, Tuple
import logging
import copy

from sglang.srt.speculative.online_structs import TTTExperience

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


class EAGLE3DecoderLayer(nn.Module):
    """
    Strictly mirroring sglang.srt.models.llama_eagle3.LlamaDecoderLayer structure.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention
        self.self_attn = nn.Module()
        # QKV Proj: input is 2 * hidden_size (concat of embed and hidden)
        self.self_attn.qkv_proj = nn.Linear(
            2 * config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False
        )
        self.self_attn.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # MLP
        if getattr(config, "model_type", "") == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(config.hidden_size, 2 * inter_size, bias=False)
        self.mlp.down_proj = nn.Linear(inter_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, embeds):
        # Corresponds to LlamaDecoderLayer.forward in llama_eagle3.py
        residual = hidden_states

        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)

        # Concat: [B, 2 * hidden]
        attn_input = torch.cat([normed_embeds, normed_hidden], dim=-1)

        # Attention Logic
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

        # Reshape [B, 1, heads * head_dim] -> [B, heads, 1, head_dim]
        B = q.shape[0]
        q = q.view(B, self.num_heads, 1, self.head_dim).transpose(1, 2)
        k = k.view(B, self.num_kv_heads, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, self.num_kv_heads, 1, self.head_dim).transpose(1, 2)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, self.num_heads * self.head_dim)

        attn_out = self.self_attn.o_proj(attn_out)

        # Residual 1
        hidden_states = residual + attn_out
        residual = hidden_states

        # MLP
        normed_hidden = self.post_attention_layernorm(hidden_states)
        gate_up = self.mlp.gate_up_proj(normed_hidden)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        mlp_out = self.mlp.down_proj(F.silu(gate) * up)

        # Final Residual
        hidden_states = residual + mlp_out
        return hidden_states


class EAGLE3InnerModel(nn.Module):
    """
    Mimics 'self.model' in LlamaForCausalLMEagle3
    """

    def __init__(self, config):
        super().__init__()
        # FC Layer (Projection)
        if hasattr(config, "target_hidden_size"):
            hidden_size_in = config.target_hidden_size
        else:
            hidden_size_in = config.hidden_size

        self.fc = nn.Linear(hidden_size_in * 3, config.hidden_size, bias=getattr(config, "bias", False))

        # Midlayer (Decoder)
        self.midlayer = EAGLE3DecoderLayer(config)

        # Final Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class TrainableEAGLE3Model(nn.Module):
    """
    Structure aligned exactly with LlamaForCausalLMEagle3 state_dict keys.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EAGLE3InnerModel(config)

        # LM Head (will be loaded/frozen later)
        vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward_step(self, hidden_states, token_embeds):
        """
        Single step forward.
        hidden_states: [B, hidden_size] (Previous State a_{t-1} or projected g_t)
        token_embeds: [B, hidden_size] (Embedding of current token)
        """
        # Note: self.model.fc is only used if input is raw g_t from target.
        # But for Step 2+, hidden_states is already internal dimension.
        # We handle FC explicitly in the training loop for Step 1.

        # Decoder Layer
        hidden_states = self.model.midlayer(hidden_states, token_embeds)

        # Final Norm
        hidden_states = self.model.norm(hidden_states)

        # Logits
        logits = self.lm_head(hidden_states)

        # Return state (pre-norm for residual connection next step?
        # Actually EAGLE passes the POST-LAYER output as next step input,
        # but before final norm.
        # In llama_eagle3.py: returns (hidden_states_to_logits, [hidden_states_to_aux])
        # hidden_states_to_aux IS the output of midlayer (before norm).
        # Wait, looking at llama_eagle3.py:
        #   hidden_states, residual = self.midlayer(...)
        #   hidden_states_to_logits, hidden_states_to_aux = self.norm(hidden_states, residual)
        #   return hidden_states_to_logits, [hidden_states_to_aux]
        # RMSNorm(x, residual) -> returns (norm(x+res), x+res)
        # So next step input should be x+res (un-normed).

        # Let's fix return values to match:
        # We need the un-normed hidden state for the next step recursion.
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

    def run(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.device)

            # 1. Initialize strictly aligned model
            model = TrainableEAGLE3Model(self.model_config.hf_config).to(self.device)

            # 2. Load Weights (Strict=False allows ignoring missing keys if any,
            # but we expect near-perfect match for critical weights)
            # Note: We filter out 'model.embed_tokens.weight' because we don't use it
            # (we receive embeddings directly).
            filtered_weights = {k: v for k, v in self.draft_weights.items() if "embed_tokens" not in k}

            missing, unexpected = model.load_state_dict(filtered_weights, strict=False)
            if missing:
                logger.info(f"[ShadowTrainer] Missing keys (expected for embeddings): {missing}")
            if unexpected:
                logger.warning(f"[ShadowTrainer] Unexpected keys: {unexpected}")

            # 3. Handle LM Head (Freeze Target Head)
            if model.lm_head.weight.shape != self.target_head_weights.shape:
                logger.warning(
                    f"[ShadowTrainer] Resizing lm_head from {model.lm_head.weight.shape} "
                    f"to {self.target_head_weights.shape}"
                )
                vocab, dim = self.target_head_weights.shape
                model.lm_head = nn.Linear(dim, vocab, bias=False).to(self.device)

            model.lm_head.weight.data.copy_(self.target_head_weights.to(self.device))
            model.lm_head.requires_grad_(False)

            # 4. Optimizer
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr
            )

            logger.info(f"[ShadowTrainer] Ready on {self.device}")

            while True:
                cmd, payload = self.data_queue.get()

                if cmd == "STOP":
                    break

                if cmd == "TRAIN_AND_SYNC":
                    experiences = payload
                    if not experiences:
                        self.result_queue.put(({}, None))
                        continue

                    model.train()
                    optimizer.zero_grad()
                    total_loss = 0.0

                    for exp in experiences:
                        exp = exp.to(self.device)

                        # Step 1: FC Project + Decode
                        # Eagle3 FC layer expects input [B, hidden*3] usually,
                        # checking llama_eagle3.py: self.fc(hidden_states)
                        # If exp.teacher_feat_g is [1, hidden], we might need to verify dimensions.
                        # Assuming exp.teacher_feat_g is already the input to FC.

                        # Apply FC only on the first step
                        current_state = model.model.fc(exp.teacher_feat_g)

                        # Decode Step 1
                        logits, current_state = model.forward_step(current_state, exp.start_token_embed)

                        # Step 2+: Autoregressive
                        # accepted_token_embeds contains embeddings for t+1, t+2...
                        # We roll out for the length of the path
                        for i in range(exp.accepted_token_embeds.size(0)):
                            next_embed = exp.accepted_token_embeds[i].unsqueeze(0)
                            logits, current_state = model.forward_step(current_state, next_embed)

                        # Final Logits vs Target Logits
                        # Target logits are for the NEXT token after the path end
                        # Wait, verification rejection point logic:
                        # If accepted_ids are [t1, t2], it means t1, t2 verified.
                        # We want to predict t3 (which might be rejected or just the end).
                        # target_logits_at_end corresponds to the distribution for t3.
                        # Our last 'logits' is the prediction for t3.

                        loss = F.kl_div(
                            F.log_softmax(logits, dim=-1),
                            F.softmax(exp.target_logits_at_end.unsqueeze(0), dim=-1),
                            reduction="batchmean"
                        )
                        loss.backward()
                        total_loss += loss.item()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    # Return weights (exclude lm_head)
                    ret_weights = {k: v.cpu() for k, v in model.state_dict().items() if "lm_head" not in k}
                    self.result_queue.put((ret_weights, total_loss / len(experiences)))

        except Exception as e:
            logger.error(f"[ShadowTrainer] Crash: {e}", exc_info=True)
            self.result_queue.put(({}, None))  # Unblock main process