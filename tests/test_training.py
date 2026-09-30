import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from coconut_reproduction.cli import main
from coconut_reproduction.training import (
    TrainConfig,
    load_model_tokenizer,
    train_epoch,
    validation_loss,
)


class ToyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, labels, **kwargs):
        labels = labels[:, 1:]
        targets = labels[labels != -100].float()
        return SimpleNamespace(loss=(self.weight - targets).square().mean())


def test_partial_accumulation_is_weighted_by_target_tokens():
    model = ToyLoss()
    loader = [{"labels": torch.tensor([[-100, 2, -100]])}, {"labels": torch.tensor([[-100, 3, 5]])}]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    loss, updates = train_epoch(model, loader, optimizer, scheduler, torch.device("cpu"), 3, False)
    # One partial accumulation group, equivalent to a loss over all three targets.
    expected_gradient = 2 * (1 - (2 + 3 + 5) / 3)
    assert model.weight.item() == pytest.approx(1 - 0.1 * expected_gradient)
    assert loss == pytest.approx((1 + 4 + 16) / 3)
    assert updates == 1


def test_validation_loss_is_token_weighted():
    loader = [{"labels": torch.tensor([[-100, 2, -100]])}, {"labels": torch.tensor([[-100, 3, 5]])}]
    assert validation_loss(ToyLoss(), loader, torch.device("cpu"), False) == pytest.approx(7)


def test_invalid_curriculum():
    with pytest.raises(ValueError, match="final stage"):
        TrainConfig(max_stage=6, total_epochs=30).validate()
    with pytest.raises(ValueError, match="local model checkpoint"):
        TrainConfig(start_stage=1).validate()


def test_newer_tokenizer_metadata_loads_in_memory(tmp_path, base_model, tokenizer):
    base_model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    path = tmp_path / "tokenizer_config.json"
    saved = json.loads(path.read_text())
    saved["extra_special_tokens"] = ["<thought>", "<bot>", "<eot>", "<ans>"]
    path.write_text(json.dumps(saved))
    original = path.read_bytes()
    _, loaded = load_model_tokenizer(str(tmp_path))
    assert loaded.convert_tokens_to_ids("<eot>") == tokenizer.convert_tokens_to_ids("<eot>")
    assert path.read_bytes() == original


def test_legacy_checkpoint_requires_explicit_plain_mode(tmp_path, base_model, legacy_tokenizer):
    base_model.resize_token_embeddings(len(legacy_tokenizer))
    base_model.save_pretrained(tmp_path)
    legacy_tokenizer.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="answer-format plain"):
        load_model_tokenizer(str(tmp_path))
    loaded, tokenizer = load_model_tokenizer(str(tmp_path), require_answer_marker=False)
    assert "<ans>" not in tokenizer.get_vocab()
    assert loaded.config.vocab_size == len(legacy_tokenizer)
    main(
        [
            "generate",
            "--checkpoint",
            str(tmp_path),
            "--answer-format",
            "plain",
            "--question",
            "Question?",
            "--max-thoughts",
            "1",
            "--max-new-tokens",
            "4",
        ]
    )


def test_train_save_load_evaluate_cli(tmp_path, base_model, tokenizer, examples):
    initial = tmp_path / "initial"
    base_model.save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    data = tmp_path / "data.json"
    data.write_text(json.dumps(examples))
    output = tmp_path / "run"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": str(initial),
                "train_data": str(data),
                "validation_data": str(data),
                "output_dir": str(output),
                "max_stage": 1,
                "epochs_per_stage": 1,
                "total_epochs": 2,
                "batch_size": 2,
                "accumulation_steps": 3,
                "warmup_steps": 0,
                "max_length": 256,
                "device": "cpu",
                "bf16": False,
            }
        )
    )
    main(["train", "--config", str(config)])
    history = json.loads((output / "history.json").read_text())
    assert [record["stage"] for record in history] == [0, 1]
    assert history[-1]["optimizer_steps"] == 2
    stage0 = output / "checkpoints/stage-0_epoch-1"
    checkpoint = output / "checkpoints/stage-1_epoch-1"
    assert (checkpoint / "model.safetensors").exists()
    assert json.loads((checkpoint / "coconut_config.json").read_text())["stage"] == 1
    result_file = tmp_path / "evaluation.json"
    main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data),
            "--output",
            str(result_file),
            "--max-thoughts",
            "1",
            "--max-new-tokens",
            "4",
        ]
    )
    result = json.loads(result_file.read_text())
    assert result["total"] == 3 and len(result["predictions"]) == 3
    assert not result["subset"]
    assert len(result["data_sha256"]) == 64
    stage0_file = tmp_path / "stage0.json"
    main(
        [
            "evaluate",
            "--checkpoint",
            str(stage0),
            "--data",
            str(data),
            "--output",
            str(stage0_file),
            "--max-new-tokens",
            "4",
            "--limit",
            "1",
        ]
    )
    result = json.loads(stage0_file.read_text())
    assert result["subset"] and result["generation"]["max_thoughts"] == 0
    main(
        [
            "generate",
            "--checkpoint",
            str(checkpoint),
            "--question",
            "Question?",
            "--max-thoughts",
            "1",
            "--max-new-tokens",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="already exists"):
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--data",
                str(data),
                "--output",
                str(result_file),
            ]
        )
    with pytest.raises(ValueError, match="new or empty"):
        main(["train", "--config", str(config)])
