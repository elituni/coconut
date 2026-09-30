"""ProsQA formatting, curriculum stages, and variable-length batch padding."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ("<thought>", "<bot>", "<eot>", "<ans>")
IGNORE_INDEX = -100


def load_examples(path: str | Path) -> list[dict]:
    """Load the shared question/answer/steps schema; retain optional graph metadata."""
    with Path(path).open(encoding="utf-8") as handle:
        examples = json.load(handle)
    if not isinstance(examples, list) or not examples:
        raise ValueError("Dataset must be a nonempty JSON list.")
    for index, item in enumerate(examples):
        if not isinstance(item, dict):
            raise ValueError(f"Example {index} must be an object.")
        for field in ("question", "answer"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Example {index} requires a nonempty {field} string.")
        if not isinstance(item.get("steps"), list) or any(
            not isinstance(step, str) or not step.strip() for step in item["steps"]
        ):
            raise ValueError(f"Example {index} requires a list of nonempty step strings.")
        if any(
            token in text
            for token in SPECIAL_TOKENS
            for text in (item["question"], item["answer"], *item["steps"])
        ):
            raise ValueError(f"Example {index} contains a reserved control token.")
    return examples


def token_ids(tokenizer, *, require_answer_marker: bool = True) -> dict[str, int]:
    """Require distinct, registered control tokens and an EOS/padding token."""
    tokens = SPECIAL_TOKENS if require_answer_marker else SPECIAL_TOKENS[:3]
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    if (
        len(set(ids.values())) != len(ids)
        or any(token not in tokenizer.get_vocab() for token in tokens)
        or tokenizer.eos_token_id is None
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("Tokenizer must register the required control tokens, EOS, and padding.")
    return ids


class ProsQADataset(Dataset):
    """Replace the first min(stage, number of steps) steps with latent placeholders.

    Question, <bot>, and latent placeholders are excluded from the language loss.
    <eot>, remaining textual steps, <ans>, the answer, and EOS are supervised.
    Overlong examples fail explicitly rather than silently truncating their answers.
    """

    def __init__(self, examples, tokenizer, stage: int = 0, max_length: int = 1024):
        if stage < 0 or max_length < 1:
            raise ValueError("Stage must be nonnegative and max_length positive.")
        self.examples = examples
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_length = max_length
        self.ids = token_ids(tokenizer)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        count = min(self.stage, len(item["steps"]))
        text = (
            f"{item['question']}\n<bot>"
            + "<thought>" * count
            + "<eot>\n"
            + "".join(f"{step}\n" for step in item["steps"][count:])
            + f"<ans>{item['answer']}{self.tokenizer.eos_token}"
        )
        ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        if len(ids) > self.max_length:
            raise ValueError(
                f"Example {index} has {len(ids)} tokens, exceeding {self.max_length}; "
                "increase max_length within the model context limit or curate the input."
            )
        eot = (ids == self.ids["<eot>"]).nonzero().flatten()
        if len(eot) != 1:
            raise ValueError(f"Example {index} must contain exactly one <eot> marker.")
        labels = ids.clone()
        labels[: eot.item()] = IGNORE_INDEX
        return {"input_ids": ids, "labels": labels, "num_thoughts": count}


class BatchCollator:
    """Right-pad only to the longest example in the batch; preserve EOS labels."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(item["input_ids"]) for item in examples)
        shape = (len(examples), length)
        ids = torch.full(shape, self.pad_token_id, dtype=torch.long)
        labels = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
        mask = torch.zeros(shape, dtype=torch.long)
        for row, item in enumerate(examples):
            size = len(item["input_ids"])
            ids[row, :size] = item["input_ids"]
            labels[row, :size] = item["labels"]
            mask[row, :size] = 1
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels,
            "num_thoughts": torch.tensor([item["num_thoughts"] for item in examples]),
        }
