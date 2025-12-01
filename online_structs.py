# python/sglang/srt/speculative/online_structs.py

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class TTTExperience:
    """
    Represents a single training experience collected during inference time for EAGLE-3 TTT.
    Transferred from Inference Process -> Shadow Training Process.
    """
    # The latent feature from the target model (Step 1 input g_t)
    # Shape: [1, hidden_size]
    # Note: Should be on CPU to avoid CUDA IPC complexity in this MVP phase
    teacher_feat_g: torch.Tensor

    # The embedding of the start token (Step 1 token input)
    # Shape: [1, hidden_size]
    start_token_embed: torch.Tensor

    # The sequence of tokens accepted by the verification (Ground Truth Path)
    # This is used to force the student to follow the teacher's path in token space,
    # while generating its own hidden states.
    accepted_ids: List[int]

    # The embeddings corresponding to accepted_ids (For Step 2+ inputs)
    # Shape: [len(accepted_ids), hidden_size]
    # We transfer embeddings to avoid loading the huge Embedding table in the shadow process.
    accepted_token_embeds: torch.Tensor

    # The target model's logits at the first rejection point (Supervision Label)
    # Shape: [vocab_size]
    target_logits_at_end: torch.Tensor

    def to(self, device: str):
        """Helper to move tensor data to a specific device"""
        return TTTExperience(
            teacher_feat_g=self.teacher_feat_g.to(device),
            start_token_embed=self.start_token_embed.to(device),
            accepted_ids=self.accepted_ids, # List has no .to()
            accepted_token_embeds=self.accepted_token_embeds.to(device),
            target_logits_at_end=self.target_logits_at_end.to(device)
        )

class OnlineExperienceBuffer:
    """
    A simple buffer to accumulate TTT experiences.
    Implements the 'accumulate and flush' strategy.
    """
    def __init__(self):
        self.buffer: List[TTTExperience] = []

    def add(self, experience: TTTExperience):
        self.buffer.append(experience)

    def get_all_and_clear(self) -> List[TTTExperience]:
        """
        Returns all accumulated experiences and clears the buffer.
        """
        data = self.buffer
        self.buffer = []
        return data

    def __len__(self):
        return len(self.buffer)