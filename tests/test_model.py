import copy

import pytest
import torch

from coconut_reproduction.data import BatchCollator, ProsQADataset, token_ids
from coconut_reproduction.model import CoconutModel


def sequential_reference(base, input_ids, bot_id, count):
    """Recompute the growing prefix per thought; no batched prefix or KV cache."""
    embeds = base.get_input_embeddings()(input_ids).clone()
    bot_position = int((input_ids[0] == bot_id).nonzero()[0])
    for offset in range(count):
        position = bot_position + 1 + offset
        output = base(
            inputs_embeds=embeds[:, :position], output_hidden_states=True, use_cache=False
        )
        replacement = output.hidden_states[-1][:, -1:, :]
        embeds = torch.cat([embeds[:, :position], replacement, embeds[:, position + 1 :]], dim=1)
    return base(inputs_embeds=embeds, use_cache=False).logits


@pytest.mark.parametrize("stage", [0, 1, 2, 6])
def test_batched_matches_sequential_logits_and_gradients(base_model, tokenizer, examples, stage):
    ids = token_ids(tokenizer)
    reference = copy.deepcopy(base_model)
    model = CoconutModel(base_model, ids["<bot>"], ids["<thought>"])
    dataset = ProsQADataset(examples, tokenizer, stage, 256)
    items = [dataset[i] for i in range(len(dataset))]
    batch = BatchCollator(tokenizer.pad_token_id)(items)
    actual = model(**batch)
    actual_loss = actual.logits.new_zeros(())
    reference_loss = actual.logits.new_zeros(())
    for row, item in enumerate(items):
        length = len(item["input_ids"])
        expected = sequential_reference(
            reference, item["input_ids"].unsqueeze(0), ids["<bot>"], item["num_thoughts"]
        )
        torch.testing.assert_close(actual.logits[row, :length], expected[0], atol=2e-6, rtol=2e-5)
        actual_loss = actual_loss + actual.logits[row, :length].square().sum()
        reference_loss = reference_loss + expected.square().sum()
    actual_loss.backward()
    reference_loss.backward()
    for parameter, expected in zip(base_model.parameters(), reference.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-4, rtol=3e-4)
    assert torch.isfinite(actual.loss)


def test_bad_marker_counts_raise(base_model, tokenizer, examples):
    ids = token_ids(tokenizer)
    model = CoconutModel(base_model, ids["<bot>"], ids["<thought>"])
    data = ProsQADataset(examples, tokenizer, 1, 256)
    batch = BatchCollator(tokenizer.pad_token_id)([data[0]])
    batch["num_thoughts"][0] = 2
    with pytest.raises(ValueError, match="placeholder"):
        model(**batch)
