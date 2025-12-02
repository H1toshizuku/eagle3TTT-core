# sglang/python/sglang/srt/speculative/eagle_utils.py
import math
from enum import IntEnum
from typing import List, Optional
import logging
import torch
import torch.nn as nn
import time
from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

logger = logging.getLogger(__name__)


if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def build_tree_efficient_native(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    draft_token_num: int,
    tree_mask_mode: int,
    bs: int,
):
    # Generate batch and token index ranges
    bs_range = torch.arange(bs, device=tree_mask.device).view(-1, 1)
    draft_token_num_range = torch.arange(draft_token_num, device=tree_mask.device)

    # Optimized common case for performance.
    if draft_token_num == 2 and topk == 1 and tree_mask_mode == TreeMaskMode.FULL_MASK:
        positions = verified_seq_len.repeat_interleave(draft_token_num)
        positions = (positions.view(bs, -1) + draft_token_num_range).view(-1)

        retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
        retrive_next_token[:, 0] = 1
        retrive_next_token[:, 1] = -1
        return (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        )

    # Precompute sequence tree indices
    draft_token_num_range1 = torch.arange(draft_token_num - 1, device=tree_mask.device)
    cum_seq_len = torch.cumsum(verified_seq_len * draft_token_num, dim=0)
    cum_seq_len = torch.cat((torch.tensor([0], device=tree_mask.device), cum_seq_len))
    cum_seq_len = cum_seq_len[:-1]
    seq_tree_idx = (
        draft_token_num * draft_token_num * torch.arange(bs, device=tree_mask.device)
        + cum_seq_len
    )

    # Batch processing for tree mask
    if tree_mask_mode == TreeMaskMode.FULL_MASK:
        token_tree_base = (
            seq_tree_idx.view(-1, 1)
            + (verified_seq_len.view(-1, 1) + draft_token_num) * draft_token_num_range
        )
        token_tree_indices = token_tree_base + verified_seq_len.view(-1, 1) + 1
    else:
        token_tree_indices = (
            bs_range * draft_token_num**2 + draft_token_num_range * draft_token_num + 1
        )

    tree_mask[token_tree_indices.flatten() - 1] = True
    indices = token_tree_indices.unsqueeze(-1) + draft_token_num_range1.view(1, 1, -1)
    tree_mask[indices.view(-1)] = False

    positions = verified_seq_len.repeat_interleave(draft_token_num)
    parent_tb_indices = selected_index // topk
    retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
    tree_mask[token_tree_indices.view(-1, 1) + draft_token_num_range1] = True

    for bid in range(bs):
        for tid in range(draft_token_num):
            position = 0
            if tid == 0:
                # Process root node
                for i in range(draft_token_num - 1, 0, -1):
                    parent_position = 0
                    parent_tb_idx = parent_tb_indices[bid][i - 1]
                    if parent_tb_idx > 0:
                        parent_token_idx = parent_list[bid][parent_tb_idx]
                        loop_num = draft_token_num - parent_position
                        for _ in range(loop_num):
                            if selected_index[bid][parent_position] == parent_token_idx:
                                parent_position += 1
                                break
                            parent_position += 1
                    if parent_position == draft_token_num:
                        continue

                    if retrive_next_token[bid][parent_position] != -1:
                        retrive_next_sibling[bid][i] = retrive_next_token[bid][
                            parent_position
                        ]
                    retrive_next_token[bid][parent_position] = i
            else:
                # Process no-root nodes
                cur_position = tid - 1
                while True:
                    position += 1
                    if cur_position >= draft_token_num:
                        tree_mask[token_tree_indices + cur_position] = True
                        parent_tb_idx = selected_index[bid][cur_position] // topk
                    else:
                        parent_tb_idx = parent_tb_indices[bid][cur_position]
                    if parent_tb_idx == 0:
                        break
                    token_idx = parent_list[bid][parent_tb_idx]
                    cur_position = 0
                    for _ in range(draft_token_num):
                        if selected_index[bid][cur_position] == token_idx:
                            break
                        cur_position += 1
                positions[bid * draft_token_num + tid] += position
    return positions, retrive_index, retrive_next_token, retrive_next_sibling, tree_mask


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        ) = build_tree_efficient_native(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            num_verify_tokens,
            tree_mask_mode,
            bs,
        )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_native(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    batch_size, num_draft_tokens = candidates.shape

    # Optimized common case for performance.
    if num_draft_tokens == 2 and accept_index.shape[1] == 2 and topk == 1:
        comparison_result = candidates[:, 1] == target_predict[:, 0]

        predicts = target_predict.flatten()

        accept_index = torch.arange(
            0, num_draft_tokens * batch_size, device=candidates.device, dtype=torch.long
        ).reshape(batch_size, num_draft_tokens)
        comparison_result = comparison_result.to(torch.int64)
        accept_index_mask = accept_index[:, 1] * comparison_result
        accept_index[:, 1] = accept_index_mask - (1 - comparison_result)

        accept_token_num = comparison_result.int()
        return predicts, accept_index, accept_token_num

    # BFS
    for bx in range(batch_size):
        cur_candidates = candidates[bx]
        cur_retrive_index = retrive_index[bx]
        cur_next_token = retrive_next_token[bx]
        cur_next_sibling = retrive_next_sibling[bx]
        cur_target = target_predict[bx]

        last_accepted_idx = cur_retrive_index[0]
        accept_index[bx, 0] = last_accepted_idx
        num_accepted = 0
        cur_node = 0

        for _ in range(1, num_draft_tokens):
            cur_node = cur_next_token[cur_node]
            found = False
            while cur_node != -1:
                draft_idx = cur_retrive_index[cur_node]
                draft_token = cur_candidates[cur_node]
                target_token = cur_target[last_accepted_idx - num_draft_tokens * bx]

                if draft_token == target_token:
                    predicts[last_accepted_idx] = target_token
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_idx
                    last_accepted_idx = draft_idx
                    found = True
                    break
                else:
                    cur_node = cur_next_sibling[cur_node]
            if not found:
                break

        accept_token_num[bx] = num_accepted
        predicts[last_accepted_idx] = cur_target[
            last_accepted_idx - num_draft_tokens * bx
        ]
    return predicts, accept_index, accept_token_num


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        predicts, accept_index, accept_token_num = verify_tree_greedy_native(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
            topk=topk,
        )

    return predicts, accept_index, accept_token_num

# --- OSD Weight Sync Utilities ---

@torch.no_grad()
def sync_shadow_to_inference_weights(shadow_model, inference_model, tp_rank, tp_size):
    """
    Sync weights from the PyTorch native Shadow Model (Full Weights) 
    to the SGLang Inference Model (Sharded/Merged Weights).
    """
    
    # 1. Sync Embedding
    if hasattr(shadow_model, "embed_tokens") and hasattr(inference_model, "embed_tokens"):
        full_embed = shadow_model.embed_tokens.weight.data
        vocab_size = full_embed.shape[0]
        shard_size = vocab_size // tp_size
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        
        target_param = inference_model.embed_tokens.weight.data
        actual_shard_size = min(end_idx, vocab_size) - start_idx
        if target_param.shape[0] == actual_shard_size:
            target_param.copy_(full_embed[start_idx:end_idx])

    # 2. Sync FC Layer
    if hasattr(shadow_model, "fc") and hasattr(inference_model, "fc"):
        full_fc = shadow_model.fc.weight.data
        out_dim = full_fc.shape[0]
        shard_size = out_dim // tp_size
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        inference_model.fc.weight.data.copy_(full_fc[start_idx:end_idx, :])

    # 3. Sync Transformer Layers
    inf_layers = None
    if hasattr(inference_model, "layers"):
        inf_layers = inference_model.layers
    elif hasattr(inference_model, "model") and hasattr(inference_model.model, "layers"):
        inf_layers = inference_model.model.layers
        
    if inf_layers is not None:
        num_layers = len(shadow_model.layers)
        for i in range(num_layers):
            shadow_layer = shadow_model.layers[i]
            inf_layer = inf_layers[i]
            
            # 3.1 Sync Self Attention
            if hasattr(shadow_layer.self_attn, "q_proj") and hasattr(inf_layer.self_attn, "qkv_proj"):
                q = shadow_layer.self_attn.q_proj.weight.data
                k = shadow_layer.self_attn.k_proj.weight.data
                v = shadow_layer.self_attn.v_proj.weight.data
                
                def get_shard(tensor, rank, size, dim=0):
                    s = tensor.shape[dim] // size
                    return tensor.narrow(dim, rank * s, s)
                
                q_shard = get_shard(q, tp_rank, tp_size)
                k_shard = get_shard(k, tp_rank, tp_size)
                v_shard = get_shard(v, tp_rank, tp_size)
                
                qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
                inf_layer.self_attn.qkv_proj.weight.data.copy_(qkv_shard)
                
                if shadow_layer.self_attn.q_proj.bias is not None and inf_layer.self_attn.qkv_proj.bias is not None:
                    q_b = shadow_layer.self_attn.q_proj.bias.data
                    k_b = shadow_layer.self_attn.k_proj.bias.data
                    v_b = shadow_layer.self_attn.v_proj.bias.data
                    q_b_shard = get_shard(q_b, tp_rank, tp_size, dim=0)
                    k_b_shard = get_shard(k_b, tp_rank, tp_size, dim=0)
                    v_b_shard = get_shard(v_b, tp_rank, tp_size, dim=0)
                    inf_layer.self_attn.qkv_proj.bias.data.copy_(torch.cat([q_b_shard, k_b_shard, v_b_shard], dim=0))

            # 3.2 Sync O_Proj
            if hasattr(shadow_layer.self_attn, "o_proj") and hasattr(inf_layer.self_attn, "o_proj"):
                full_o = shadow_layer.self_attn.o_proj.weight.data
                in_dim = full_o.shape[1]
                shard_size = in_dim // tp_size
                start_idx = tp_rank * shard_size
                end_idx = (tp_rank + 1) * shard_size
                inf_layer.self_attn.o_proj.weight.data.copy_(full_o[:, start_idx:end_idx])

            # 3.3 Sync MLP (GateUp)
            if hasattr(shadow_layer.mlp, "gate_proj") and hasattr(inf_layer.mlp, "gate_up_proj"):
                gate = shadow_layer.mlp.gate_proj.weight.data
                up = shadow_layer.mlp.up_proj.weight.data
                
                def get_shard(tensor, rank, size):
                    s = tensor.shape[0] // size
                    return tensor.narrow(0, rank * s, s)
                
                gate_shard = get_shard(gate, tp_rank, tp_size)
                up_shard = get_shard(up, tp_rank, tp_size)
                
                gate_up_shard = torch.cat([gate_shard, up_shard], dim=0)
                inf_layer.mlp.gate_up_proj.weight.data.copy_(gate_up_shard)

            # 3.4 Sync MLP (Down)
            if hasattr(shadow_layer.mlp, "down_proj") and hasattr(inf_layer.mlp, "down_proj"):
                full_down = shadow_layer.mlp.down_proj.weight.data
                in_dim = full_down.shape[1]
                shard_size = in_dim // tp_size
                start_idx = tp_rank * shard_size
                end_idx = (tp_rank + 1) * shard_size
                inf_layer.mlp.down_proj.weight.data.copy_(full_down[:, start_idx:end_idx])
                
            # 3.5 Sync Norms
            if hasattr(shadow_layer, "input_layernorm") and hasattr(inf_layer, "input_layernorm"):
                inf_layer.input_layernorm.weight.data.copy_(shadow_layer.input_layernorm.weight.data)
            if hasattr(shadow_layer, "post_attention_layernorm") and hasattr(inf_layer, "post_attention_layernorm"):
                inf_layer.post_attention_layernorm.weight.data.copy_(shadow_layer.post_attention_layernorm.weight.data)

    # 4. Sync Final Norm
    if hasattr(shadow_model, "norm") and hasattr(inference_model, "norm"):
        inference_model.norm.weight.data.copy_(shadow_model.norm.weight.data)

    # 5. Sync LM Head
    if hasattr(shadow_model, "lm_head") and hasattr(inference_model, "lm_head"):
        full_head = shadow_model.lm_head.weight.data
        out_dim = full_head.shape[0]
        shard_size = out_dim // tp_size
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        inference_model.lm_head.weight.data.copy_(full_head[start_idx:end_idx, :])


def safe_copy(src_mod, dst_mod, name_hint=""):
    if not isinstance(src_mod, nn.Module) or not isinstance(dst_mod, nn.Module):
        return
    if not hasattr(src_mod, "weight") or not hasattr(dst_mod, "weight"):
        return

    # [新增] 采样打印 (仅针对 fc 层，且只打印一次或偶尔打印)
    if "fc" in name_hint: #and torch.rand(1).item() < 0.1:  # 10% 概率打印，防止刷屏
        val_src = src_mod.weight.data[0, 0].item()
        val_dst_before = dst_mod.weight.data[0, 0].item()
    
    dst_mod.weight.data.copy_(src_mod.weight.data)  # 执行复制

    if hasattr(src_mod, "bias") and src_mod.bias is not None and hasattr(dst_mod, "bias") and dst_mod.bias is not None:
        dst_mod.bias.data.copy_(src_mod.bias.data)

    # [新增] 打印结果
    if "fc" in name_hint and 'val_src' in locals():
        val_dst_after = dst_mod.weight.data[0, 0].item()
        logger.info(f"OSD Sync [{name_hint}]: Src={val_src:.6f} -> Dst_Before={val_dst_before:.6f} -> Dst_After={val_dst_after:.6f}")


@torch.no_grad()
def sync_shadow_to_inference_weights(shadow_model, inference_model, tp_rank, tp_size):
    """
    Sync weights from the PyTorch native Shadow Model (Full Weights) 
    to the SGLang Inference Model (Sharded/Merged Weights).
    """
    # === [DEBUG] Fingerprint Verification to File ===
    log_file = "/root/autodl-tmp/logs/osd_sync_debug.log"
    probe_val_shadow = 0.0
    probe_val_inf_before = 0.0
    
    if hasattr(shadow_model, "fc") and hasattr(inference_model, "fc"):
        probe_val_shadow = shadow_model.fc.weight.data[0,0].item()
        probe_val_inf_before = inference_model.fc.weight.data[0,0].item()
    
    # # Perform Sync (Existing Logic)
    # # 1. Sync Embedding (Skip if frozen, but syncing makes sure they stay same just in case)
    # inf_embed = None
    # if hasattr(inference_model, "model") and hasattr(inference_model.model, "embed_tokens"):
    #     inf_embed = inference_model.model.embed_tokens
    # elif hasattr(inference_model, "embed_tokens"):
    #     inf_embed = inference_model.embed_tokens
    
    # if hasattr(shadow_model, "embed_tokens"):
    #     safe_copy(shadow_model.embed_tokens, inf_embed, "embed_tokens")

    # 2. Sync FC
    inf_fc = None
    if hasattr(inference_model, "model") and hasattr(inference_model.model, "fc"):
        inf_fc = inference_model.model.fc
    elif hasattr(inference_model, "fc"):
        inf_fc = inference_model.fc

    if hasattr(shadow_model, "fc"):
        safe_copy(shadow_model.fc, inf_fc, "fc")

    # 3. Sync Transformer Layers
    inf_layers = None
    if hasattr(inference_model, "layers"):
        inf_layers = inference_model.layers
    elif hasattr(inference_model, "model") and hasattr(inference_model.model, "layers"):
        inf_layers = inference_model.model.layers
        
    if inf_layers is not None:
        num_layers = len(shadow_model.layers)
        for i in range(num_layers):
            shadow_layer = shadow_model.layers[i]
            inf_layer = inf_layers[i]
            
            # 3.1 Self Attention (QKV)
            if hasattr(shadow_layer.self_attn, "q_proj") and hasattr(inf_layer.self_attn, "qkv_proj") and isinstance(inf_layer.self_attn.qkv_proj, nn.Module):
                q = shadow_layer.self_attn.q_proj.weight.data
                k = shadow_layer.self_attn.k_proj.weight.data
                v = shadow_layer.self_attn.v_proj.weight.data
                
                def get_shard(tensor, rank, size, dim=0):
                    s = tensor.shape[dim] // size
                    return tensor.narrow(dim, rank * s, s)
                
                q_shard = get_shard(q, tp_rank, tp_size)
                k_shard = get_shard(k, tp_rank, tp_size)
                v_shard = get_shard(v, tp_rank, tp_size)
                
                qkv_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)
                
                if inf_layer.self_attn.qkv_proj.weight.shape == qkv_shard.shape:
                     inf_layer.self_attn.qkv_proj.weight.data.copy_(qkv_shard)
                
                if shadow_layer.self_attn.q_proj.bias is not None and inf_layer.self_attn.qkv_proj.bias is not None:
                    q_b = shadow_layer.self_attn.q_proj.bias.data
                    k_b = shadow_layer.self_attn.k_proj.bias.data
                    v_b = shadow_layer.self_attn.v_proj.bias.data
                    q_b_shard = get_shard(q_b, tp_rank, tp_size, dim=0)
                    k_b_shard = get_shard(k_b, tp_rank, tp_size, dim=0)
                    v_b_shard = get_shard(v_b, tp_rank, tp_size, dim=0)
                    inf_layer.self_attn.qkv_proj.bias.data.copy_(torch.cat([q_b_shard, k_b_shard, v_b_shard], dim=0))

            # 3.2 O_Proj
            if hasattr(shadow_layer.self_attn, "o_proj") and hasattr(inf_layer.self_attn, "o_proj"):
                safe_copy(shadow_layer.self_attn.o_proj, inf_layer.self_attn.o_proj, f"layer_{i}_o")

            # 3.3 MLP (GateUp)
            if hasattr(shadow_layer.mlp, "gate_proj") and hasattr(inf_layer.mlp, "gate_up_proj") and isinstance(inf_layer.mlp.gate_up_proj, nn.Module):
                gate = shadow_layer.mlp.gate_proj.weight.data
                up = shadow_layer.mlp.up_proj.weight.data
                
                def get_shard(tensor, rank, size):
                    s = tensor.shape[0] // size
                    return tensor.narrow(0, rank * s, s)
                
                gate_shard = get_shard(gate, tp_rank, tp_size)
                up_shard = get_shard(up, tp_rank, tp_size)
                
                gate_up_shard = torch.cat([gate_shard, up_shard], dim=0)
                if inf_layer.mlp.gate_up_proj.weight.shape == gate_up_shard.shape:
                    inf_layer.mlp.gate_up_proj.weight.data.copy_(gate_up_shard)

            # 3.4 MLP (Down)
            if hasattr(shadow_layer.mlp, "down_proj") and hasattr(inf_layer.mlp, "down_proj"):
                safe_copy(shadow_layer.mlp.down_proj, inf_layer.mlp.down_proj, f"layer_{i}_down")
                
            # 3.5 Norms
            if hasattr(shadow_layer, "input_layernorm") and hasattr(inf_layer, "input_layernorm"):
                safe_copy(shadow_layer.input_layernorm, inf_layer.input_layernorm, f"layer_{i}_input_norm")
            if hasattr(shadow_layer, "post_attention_layernorm") and hasattr(inf_layer, "post_attention_layernorm"):
                safe_copy(shadow_layer.post_attention_layernorm, inf_layer.post_attention_layernorm, f"layer_{i}_post_attn_norm")

    # 4. Final Norm
    inf_norm = None
    if hasattr(inference_model, "model") and hasattr(inference_model.model, "norm"):
        inf_norm = inference_model.model.norm
    elif hasattr(inference_model, "norm"):
        inf_norm = inference_model.norm

    if hasattr(shadow_model, "norm") and inf_norm is not None:
        safe_copy(shadow_model.norm, inf_norm, "final_norm")

    # # 5. LM Head
    # inf_head = None
    # if hasattr(inference_model, "lm_head"):
    #     inf_head = inference_model.lm_head

    # if hasattr(shadow_model, "lm_head") and inf_head is not None:
    #     safe_copy(shadow_model.lm_head, inf_head, "lm_head")

    # === [DEBUG] Post Check ===
    if hasattr(inference_model, "fc"):
        probe_val_inf_after = inference_model.fc.weight.data[0,0].item()
    elif hasattr(inference_model, "model") and hasattr(inference_model.model, "fc"):
        probe_val_inf_after = inference_model.model.fc.weight.data[0,0].item()
    else:
        probe_val_inf_after = -3.0
        
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] Sync Check: Shadow={probe_val_shadow:.6f} -> Inf_Before={probe_val_inf_before:.6f} -> Inf_After={probe_val_inf_after:.6f}\n"
    try:
        with open(log_file, "a") as f:
            f.write(msg)
    except:
        pass


@torch.no_grad()
def sync_inference_to_shadow_weights(inference_model, shadow_model, tp_rank, tp_size):
    """
    Inverse operation: Sync weights FROM SGLang Inference Model TO Shadow Model.
    Used for initialization.
    """
    if tp_size > 1:
        logger.warning("OSD: TP > 1 not supported for weight sync yet.")
        return

    # 1. Sync Embedding
    inf_embed = None
    if hasattr(inference_model, "model") and hasattr(inference_model.model, "embed_tokens"):
        inf_embed = inference_model.model.embed_tokens
    elif hasattr(inference_model, "embed_tokens"):
        inf_embed = inference_model.embed_tokens
    
    if hasattr(shadow_model, "embed_tokens") and inf_embed is not None:
        safe_copy(inf_embed, shadow_model.embed_tokens, "embed_tokens")

    # 2. Sync FC
    inf_fc = None
    if hasattr(inference_model, "model") and hasattr(inference_model.model, "fc"):
        inf_fc = inference_model.model.fc
    elif hasattr(inference_model, "fc"):
        inf_fc = inference_model.fc

    if hasattr(shadow_model, "fc") and inf_fc is not None:
        safe_copy(inf_fc, shadow_model.fc, "fc")

    # 3. Sync Transformer Layers
    inf_layers = None
    if hasattr(inference_model, "layers"):
        inf_layers = inference_model.layers
    elif hasattr(inference_model, "model") and hasattr(inference_model.model, "layers"):
        inf_layers = inference_model.model.layers

    if inf_layers is not None:
        num_layers = len(shadow_model.layers)
        for i in range(num_layers):
            shadow_layer = shadow_model.layers[i]
            inf_layer = inf_layers[i]

            # 3.1 Self Attention (QKV)
            if hasattr(shadow_layer.self_attn, "q_proj") and hasattr(inf_layer.self_attn, "qkv_proj") and isinstance(inf_layer.self_attn.qkv_proj, nn.Module):
                qkv_weight = inf_layer.self_attn.qkv_proj.weight.data
                
                hidden_size = shadow_model.config.hidden_size
                num_heads = shadow_model.config.num_attention_heads
                num_kv_heads = shadow_model.config.num_key_value_heads
                head_dim = hidden_size // num_heads
                
                q_dim = num_heads * head_dim
                k_dim = num_kv_heads * head_dim
                v_dim = num_kv_heads * head_dim
                
                if qkv_weight.shape[0] == q_dim + k_dim + v_dim:
                    q_w, k_w, v_w = torch.split(qkv_weight, [q_dim, k_dim, v_dim], dim=0)
                    
                    shadow_layer.self_attn.q_proj.weight.data.copy_(q_w)
                    shadow_layer.self_attn.k_proj.weight.data.copy_(k_w)
                    shadow_layer.self_attn.v_proj.weight.data.copy_(v_w)

                    if inf_layer.self_attn.qkv_proj.bias is not None:
                        qkv_bias = inf_layer.self_attn.qkv_proj.bias.data
                        q_b, k_b, v_b = torch.split(qkv_bias, [q_dim, k_dim, v_dim], dim=0)
                        if shadow_layer.self_attn.q_proj.bias is not None:
                            shadow_layer.self_attn.q_proj.bias.data.copy_(q_b)
                        if shadow_layer.self_attn.k_proj.bias is not None:
                            shadow_layer.self_attn.k_proj.bias.data.copy_(k_b)
                        if shadow_layer.self_attn.v_proj.bias is not None:
                            shadow_layer.self_attn.v_proj.bias.data.copy_(v_b)

            # 3.2 O_Proj
            if hasattr(shadow_layer.self_attn, "o_proj") and hasattr(inf_layer.self_attn, "o_proj"):
                safe_copy(inf_layer.self_attn.o_proj, shadow_layer.self_attn.o_proj, f"layer_{i}_o")

            # 3.3 MLP (GateUp)
            if hasattr(shadow_layer.mlp, "gate_proj") and hasattr(inf_layer.mlp, "gate_up_proj") and isinstance(inf_layer.mlp.gate_up_proj, nn.Module):
                gate_up_weight = inf_layer.mlp.gate_up_proj.weight.data
                intermediate_size = shadow_model.config.intermediate_size
                
                if gate_up_weight.shape[0] == 2 * intermediate_size:
                    gate_w, up_w = torch.split(gate_up_weight, intermediate_size, dim=0)
                    shadow_layer.mlp.gate_proj.weight.data.copy_(gate_w)
                    shadow_layer.mlp.up_proj.weight.data.copy_(up_w)

            # 3.4 MLP (Down)
            if hasattr(shadow_layer.mlp, "down_proj") and hasattr(inf_layer.mlp, "down_proj"):
                safe_copy(inf_layer.mlp.down_proj, shadow_layer.mlp.down_proj, f"layer_{i}_down")

            # 3.5 Norms
            if hasattr(shadow_layer, "input_layernorm") and hasattr(inf_layer, "input_layernorm"):
                safe_copy(inf_layer.input_layernorm, shadow_layer.input_layernorm, f"layer_{i}_input_norm")
            if hasattr(shadow_layer, "post_attention_layernorm") and hasattr(inf_layer, "post_attention_layernorm"):
                safe_copy(inf_layer.post_attention_layernorm, shadow_layer.post_attention_layernorm, f"layer_{i}_post_attn_norm")

    # 4. Final Norm
    inf_norm = None
    if hasattr(inference_model, "model") and hasattr(inference_model.model, "norm"):
        inf_norm = inference_model.model.norm
    elif hasattr(inference_model, "norm"):
        inf_norm = inference_model.norm

    if hasattr(shadow_model, "norm") and inf_norm is not None:
        safe_copy(inf_norm, shadow_model.norm, "final_norm")

    # 5. LM Head
    inf_head = None
    if hasattr(inference_model, "lm_head"):
        inf_head = inference_model.lm_head

    if hasattr(shadow_model, "lm_head") and inf_head is not None:
        safe_copy(inf_head, shadow_model.lm_head, "lm_head")