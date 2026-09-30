import json

import pytest
import torch

from coconut_reproduction.data import BatchCollator, ProsQADataset, load_examples, token_ids


@pytest.mark.parametrize("stage", [0, 1, 6])
def test_curriculum_masking_and_padding(tokenizer, examples, stage):
    dataset = ProsQADataset(examples, tokenizer, stage, max_length=256)
    items = [dataset[i] for i in range(len(dataset))]
    ids = token_ids(tokenizer)
    for item, raw in zip(items, examples):
        assert item["num_thoughts"] == min(stage, len(raw["steps"]))
        assert int((item["input_ids"] == ids["<thought>"]).sum()) == item["num_thoughts"]
        eot = int((item["input_ids"] == ids["<eot>"]).nonzero()[0])
        assert (item["labels"][:eot] == -100).all()
        assert torch.equal(item["labels"][eot:], item["input_ids"][eot:])
        assert int(item["labels"][-1]) == tokenizer.eos_token_id
    batch = BatchCollator(tokenizer.pad_token_id)(items)
    assert batch["input_ids"].shape[1] == max(len(item["input_ids"]) for item in items)
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    for row, item in enumerate(items):
        assert batch["labels"][row, len(item["input_ids"]) - 1] == tokenizer.eos_token_id


def test_overlong_example_fails_instead_of_dropping_answer(tokenizer, examples):
    with pytest.raises(ValueError, match="exceeding"):
        ProsQADataset(examples, tokenizer, max_length=4)[0]


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        [{"question": "q", "answer": "a"}],
        [{"question": "q<bot>", "answer": "a", "steps": []}],
    ],
)
def test_invalid_schema(tmp_path, value):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_examples(path)


def test_schema_preserves_optional_metadata(tmp_path, examples):
    examples[0]["edges"] = [[0, 1]]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(examples))
    assert load_examples(path) == examples
