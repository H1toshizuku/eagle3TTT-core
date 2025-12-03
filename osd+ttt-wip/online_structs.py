import torch
from dataclasses import dataclass,field
from typing import List, Optional, Dict
from torch.nn.utils.rnn import pad_sequence


@dataclass
class TTTExperience:
    """
    Represents a single training experience collected during inference time for EAGLE-3 TTT.
    Transferred from Inference Process -> Shadow Training Process.
    """
    # The latent feature from the target model (Step 1 input g_t)
    # Shape: [1, hidden_size]
    # Note: Kept on CPU initially to avoid CUDA IPC complexity during transfer
    teacher_feat_g: torch.Tensor

    # The embedding of the start token (Step 1 token input)
    # Shape: [1, hidden_size]
    start_token_embed: torch.Tensor

    # The embeddings corresponding to accepted tokens (For Step 2+ inputs)
    # Shape: [len(accepted_ids), hidden_size]
    # We transfer embeddings instead of IDs to avoid loading the huge Embedding table in the shadow process.
    accepted_token_embeds: torch.Tensor

    # The target model's logits at the supervision point (usually the first rejection point or end of tree)
    # Shape: [vocab_size]
    target_logits_at_end: torch.Tensor

    # The absolute position ID of the start token.
    # Essential for correct RoPE calculation in the shadow model.
    # Shape: [1]
    start_position_id: torch.Tensor

    # Debug info
    accepted_ids: List[int] = field(default_factory=list)

    def to(self, device: str):
        """Helper to move tensor data to a specific device"""
        return TTTExperience(
            teacher_feat_g=self.teacher_feat_g.to(device),
            start_token_embed=self.start_token_embed.to(device),
            accepted_token_embeds=self.accepted_token_embeds.to(device),
            target_logits_at_end=self.target_logits_at_end.to(device),
            start_position_id=self.start_position_id.to(device),
            accepted_ids=self.accepted_ids
        )


class OnlineExperienceBuffer:
    """
    A simple thread-safe buffer to accumulate TTT experiences.
    Implements the 'accumulate and flush' strategy.
    """

    def __init__(self, capacity: int = 2048):
        self.buffer: List[TTTExperience] = []
        self.capacity = capacity

    def add(self, experience: TTTExperience):
        if len(self.buffer) < self.capacity:
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


def collate_ttt_experiences(experiences: List[TTTExperience], device: str) -> Optional[Dict[str, torch.Tensor]]:
    """
    Batches a list of TTTExperience objects into tensors for training.
    Handles padding for variable length accepted sequences.
    """
    if not experiences:
        return None

    # 1. Stack fixed-size tensors
    # teacher_feat_g: [Batch, Hidden]
    teacher_feat_g = torch.cat([e.teacher_feat_g for e in experiences], dim=0).to(device)

    # start_token_embed: [Batch, Hidden]
    start_token_embed = torch.cat([e.start_token_embed for e in experiences], dim=0).to(device)

    # target_logits: [Batch, Vocab]
    target_logits = torch.stack([e.target_logits_at_end for e in experiences], dim=0).to(device)

    # start_position_ids: [Batch] - Ensure it's 1D for RoPE broadcasting
    start_positions = torch.cat([e.start_position_id for e in experiences], dim=0).view(-1).to(device)

    # 2. Pad variable-length accepted sequences
    # sequences: List of [L, Hidden] tensors
    sequences = [e.accepted_token_embeds.to(device) for e in experiences]

    # Record original lengths for masking in the training loop
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long, device=device)

    # Handle padding
    # pad_sequence output: [Batch, MaxLen, Hidden] (because batch_first=True)
    if len(sequences) > 0 and lengths.max() > 0:
        padded_embeds = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    else:
        # Handle case where all experiences have 0 accepted tokens (only Step 1 training)
        # Create an empty tensor [Batch, 0, Hidden]
        padded_embeds = torch.zeros((len(experiences), 0, teacher_feat_g.shape[1]), device=device)

    return {
        "teacher_feat_g": teacher_feat_g,
        "start_token_embed": start_token_embed,
        "target_logits_at_end": target_logits,
        "start_position_ids": start_positions,
        "padded_embeds": padded_embeds,
        "lengths": lengths
    }