"""Batched hidden-state feedback for GPT-2, with gradients through latent thoughts."""

import torch
from torch import nn
from transformers import GPT2LMHeadModel


class CoconutModel(nn.Module):
    """Compute continuous thoughts with a KV cache, then inject them into a full pass.

    Rows may have different question lengths and latent counts. The cached prefix
    masks positions beyond each row's <bot>; explicit position IDs keep recurrent
    thoughts aligned despite those masked cache slots. The final pass recomputes
    the sequence to provide a single causal language loss, as in the notebook.
    """

    def __init__(self, base_model: GPT2LMHeadModel, bot_id: int, thought_id: int):
        super().__init__()
        if not isinstance(base_model, GPT2LMHeadModel):
            raise TypeError("This implementation supports GPT2LMHeadModel only.")
        self.base_model = base_model
        self.bot_id = bot_id
        self.thought_id = thought_id

    def forward(self, input_ids, attention_mask, labels, num_thoughts):
        if num_thoughts.ndim != 1 or len(num_thoughts) != len(input_ids):
            raise ValueError("num_thoughts must contain one count per batch row.")
        if (num_thoughts < 0).any():
            raise ValueError("Latent counts must be nonnegative.")
        if ((input_ids == self.bot_id).sum(dim=1) != 1).any():
            raise ValueError("Each sequence must contain exactly one <bot> token.")
        if ((input_ids == self.thought_id).sum(dim=1) != num_thoughts).any():
            raise ValueError("Latent counts must match the placeholder tokens.")
        max_count = int(num_thoughts.max())
        if max_count == 0:
            return self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False
            )

        batch_size = input_ids.size(0)
        device = input_ids.device
        rows = torch.arange(batch_size, device=device)
        bot_positions = (input_ids == self.bot_id).long().argmax(dim=1)
        prefix_length = int(bot_positions.max()) + 1
        if prefix_length + max_count - 1 > self.base_model.config.n_positions:
            raise ValueError(
                "The padded latent cache exceeds the context window for this mixed-length batch; "
                "reduce batch size or group examples with similar question lengths."
            )
        prefix_positions = torch.arange(prefix_length, device=device).unsqueeze(0)
        prefix_mask = attention_mask[:, :prefix_length] * (
            prefix_positions <= bot_positions.unsqueeze(1)
        )
        outputs = self.base_model(
            input_ids=input_ids[:, :prefix_length],
            attention_mask=prefix_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        thought = outputs.hidden_states[-1][rows, bot_positions].unsqueeze(1)
        thoughts = [thought]
        cache = outputs.past_key_values
        cache_mask = prefix_mask
        for offset in range(1, max_count):
            cache_mask = torch.cat(
                [cache_mask, torch.ones((batch_size, 1), dtype=cache_mask.dtype, device=device)],
                dim=1,
            )
            outputs = self.base_model(
                inputs_embeds=thought,
                attention_mask=cache_mask,
                position_ids=(bot_positions + offset).unsqueeze(1),
                past_key_values=cache,
                output_hidden_states=True,
                use_cache=True,
            )
            thought = outputs.hidden_states[-1][:, -1:, :]
            thoughts.append(thought)
            cache = outputs.past_key_values

        offsets = torch.arange(max_count, device=device).unsqueeze(0)
        active = offsets < num_thoughts.unsqueeze(1)
        positions = bot_positions.unsqueeze(1) + 1 + offsets
        active_rows = rows.unsqueeze(1).expand(batch_size, max_count)[active]
        active_positions = positions[active]
        if (input_ids[active_rows, active_positions] != self.thought_id).any():
            raise ValueError("Thought placeholders must immediately follow <bot>.")
        embeds = self.base_model.get_input_embeddings()(input_ids).clone()
        embeds[active_rows, active_positions] = torch.cat(thoughts, dim=1).to(embeds.dtype)[active]
        return self.base_model(
            inputs_embeds=embeds, attention_mask=attention_mask, labels=labels, use_cache=False
        )
