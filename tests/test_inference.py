from types import SimpleNamespace

import pytest
import torch
from torch import nn

from coconut_reproduction.data import token_ids
from coconut_reproduction.inference import extract_answer, generate


class ScriptedModel(nn.Module):
    """Predict <eot> after a chosen number of consumed latent embeddings."""

    def __init__(self, tokenizer, stop_after):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(n_positions=256)
        self.ids = token_ids(tokenizer)
        self.eos = tokenizer.eos_token_id
        self.vocab = len(tokenizer)
        self.stop_after = stop_after
        self.latents = 0
        self.suffix = None
        self.in_text = False

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            self.latents += 1
        if input_ids is not None and int(input_ids[0, 0]) == self.ids["<eot>"]:
            self.suffix = input_ids[0].tolist()
            self.in_text = True
            next_id = self.ids["<ans>"]
        elif self.in_text:
            next_id = self.eos
        else:
            next_id = (
                self.ids["<eot>"] if self.latents >= self.stop_after else self.ids["<thought>"]
            )
        logits = torch.full((1, 1, self.vocab), -100.0)
        logits[0, 0, next_id] = 1.0
        return SimpleNamespace(
            logits=logits, hidden_states=(torch.ones(1, 1, 8),), past_key_values=None
        )


@pytest.mark.parametrize(
    "stop_after,cap,expected,stopped",
    [(0, 12, 0, True), (1, 12, 1, True), (3, 12, 3, True), (99, 2, 2, False), (99, 0, 0, False)],
)
def test_adaptive_counts_actual_thoughts(tokenizer, stop_after, cap, expected, stopped):
    model = ScriptedModel(tokenizer, stop_after)
    result = generate(model, tokenizer, "Question?", max_thoughts=cap)
    assert result.num_thoughts == model.latents == expected
    assert result.stopped_on_eot == stopped
    assert model.suffix == [token_ids(tokenizer)["<eot>"], *tokenizer.encode("\n")]
    assert result.ended_on_eos


def test_fixed_policy_ignores_predicted_stop(tokenizer):
    model = ScriptedModel(tokenizer, stop_after=0)
    result = generate(model, tokenizer, "Question?", max_thoughts=3, latent_policy="fixed")
    assert result.num_thoughts == 3
    assert not result.stopped_on_eot


def test_answer_extraction_requires_separator():
    assert extract_answer("reasoning\n<ans> YES. <eos>", "<eos>") == "yes."
    assert extract_answer("yes.", "<eos>") == ""
    assert extract_answer(" YES. <eos>", "<eos>", "plain") == "yes."
    assert extract_answer("reasoning\nyes.<eos>", "<eos>", "plain") == "reasoning\nyes."


def test_explicit_legacy_separator(tokenizer):
    model = ScriptedModel(tokenizer, stop_after=0)
    generate(model, tokenizer, "Question?", max_thoughts=0, eot_newline=False)
    assert model.suffix == [token_ids(tokenizer)["<eot>"]]


def test_context_budget_guard(base_model, tokenizer):
    with pytest.raises(ValueError, match="context window"):
        generate(base_model, tokenizer, "Question?", max_new_tokens=500)


def test_real_gpt2_generation(base_model, tokenizer):
    result = generate(
        base_model, tokenizer, "Question?", max_thoughts=2, max_new_tokens=4, latent_policy="fixed"
    )
    assert result.num_thoughts == 2
    assert isinstance(result.text, str)
