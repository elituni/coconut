"""Seeded curriculum training with token-weighted accumulation and saved run records."""

import hashlib
import json
import math
import platform
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

from .data import (
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    BatchCollator,
    ProsQADataset,
    load_examples,
    token_ids,
)
from .model import CoconutModel


@dataclass
class TrainConfig:
    model: str = "openai-community/gpt2"
    train_data: str = "data/prosqa_train.json"
    validation_data: str = "data/prosqa_valid.json"
    output_dir: str = "runs/prosqa"
    max_stage: int = 6
    epochs_per_stage: int = 5
    total_epochs: int = 50
    batch_size: int = 32
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_length: int = 1024
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    bf16: bool = True
    start_stage: int = 0
    completed_epochs: int = 0
    train_limit: int | None = None
    validation_limit: int | None = None

    def validate(self):
        for field in (
            "epochs_per_stage",
            "total_epochs",
            "batch_size",
            "accumulation_steps",
            "max_length",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive.")
        if not 0 <= self.start_stage <= self.max_stage:
            raise ValueError("Require 0 <= start_stage <= max_stage.")
        if self.total_epochs <= self.max_stage * self.epochs_per_stage:
            raise ValueError("total_epochs must leave at least one epoch for the final stage.")
        if self.num_workers < 0 or self.warmup_steps < 0 or self.completed_epochs < 0:
            raise ValueError("Worker, warmup, and completed-epoch counts must be nonnegative.")
        if self.completed_epochs > stage_epochs(self, self.start_stage):
            raise ValueError("completed_epochs exceeds the first stage's epoch count.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
        for limit in (self.train_limit, self.validation_limit):
            if limit is not None and limit < 1:
                raise ValueError("Dataset limits must be positive.")
        if (self.start_stage or self.completed_epochs) and not Path(self.model).is_dir():
            raise ValueError("Continuing a curriculum requires a local model checkpoint.")


def stage_epochs(config: TrainConfig, stage: int) -> int:
    return (
        config.epochs_per_stage
        if stage < config.max_stage
        else config.total_epochs - config.max_stage * config.epochs_per_stage
    )


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_device(name: str) -> torch.device:
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else (torch.device(name))
    )


def autocast_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def load_model_tokenizer(
    model_path: str, *, initialize: bool = False, require_answer_marker: bool = True
):
    """Public GPT-2 needs no token file. HF_TOKEN is handled by Hugging Face itself."""
    tokenizer_options = {}
    tokenizer_config = Path(model_path) / "tokenizer_config.json"
    if tokenizer_config.is_file():
        saved = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        # Newer Transformers saves a list here; 4.51 expects a name-to-token map.
        # Normalize in memory and retain the original checkpoint on disk.
        extra = saved.get("extra_special_tokens")
        if isinstance(extra, list):
            tokenizer_options = {"extra_special_tokens": {}, "additional_special_tokens": extra}
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_options)
    if not initialize and require_answer_marker and "<ans>" not in tokenizer.get_vocab():
        raise ValueError(
            "This checkpoint uses the older format without <ans>. "
            "Use --answer-format plain for explicit legacy inference. "
            "Adding a new token would change the trained model."
        )
    base = GPT2LMHeadModel.from_pretrained(
        model_path, attn_implementation="eager", use_safetensors=True
    )
    if initialize:
        tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
        tokenizer.pad_token = tokenizer.eos_token
        base.resize_token_embeddings(len(tokenizer))
    token_ids(tokenizer, require_answer_marker=require_answer_marker)
    return base, tokenizer


def target_count(batch) -> int:
    """The causal LM shifts labels by one; count only actual loss targets."""
    return int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())


def train_epoch(model, loader, optimizer, scheduler, device, accumulation_steps, use_bf16):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    group_tokens = 0
    loss_sum = 0.0
    total_tokens = 0
    updates = 0
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        batch = {key: value.to(device) for key, value in batch.items()}
        tokens = target_count(batch)
        if tokens == 0:
            raise ValueError("Training batch contains no supervised next-token targets.")
        with autocast_context(device, use_bf16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss; no checkpoint saved for this epoch.")
        # Sum target-token losses now, normalize the accumulated gradient at the update.
        # This correctly handles partial groups and batches with different token counts.
        (loss * tokens).backward()
        group_tokens += tokens
        loss_sum += loss.item() * tokens
        total_tokens += tokens
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_tokens)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            group_tokens = 0
            updates += 1
    return loss_sum / total_tokens, updates


@torch.no_grad()
def validation_loss(model, loader, device, use_bf16):
    model.eval()
    loss_sum = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        tokens = target_count(batch)
        if tokens == 0:
            raise ValueError("Validation batch contains no supervised next-token targets.")
        with autocast_context(device, use_bf16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss.")
        loss_sum += loss.item() * tokens
        total_tokens += tokens
    return loss_sum / total_tokens


def train(config: TrainConfig):
    config.validate()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = select_device(config.device)
    use_bf16 = config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    train_data = load_examples(config.train_data)[: config.train_limit]
    validation_data = load_examples(config.validation_data)[: config.validation_limit]
    base, tokenizer = load_model_tokenizer(
        config.model, initialize=config.start_stage == 0 and config.completed_epochs == 0
    )
    if config.max_length > base.config.n_positions:
        raise ValueError("max_length exceeds the model's position-embedding limit.")
    ids = token_ids(tokenizer)
    model = CoconutModel(base.to(device), ids["<bot>"], ids["<thought>"])
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output_dir must be new or empty to preserve existing run records.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "bf16_enabled": use_bf16,
        "dataset_sizes": {"train": len(train_data), "validation": len(validation_data)},
        "dataset_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("train", config.train_data), ("validation", config.validation_data))
        },
        "continuation": "weights only; optimizer and scheduler reset at each stage",
    }
    write_json(output / "run.json", metadata)
    history = []
    optimizer_steps = 0
    for stage in range(config.start_stage, config.max_stage + 1):
        completed = config.completed_epochs if stage == config.start_stage else 0
        epochs = stage_epochs(config, stage)
        if completed == epochs:
            continue
        generator = torch.Generator().manual_seed(config.seed + stage)
        loaders = []
        for data, shuffle in ((train_data, True), (validation_data, False)):
            loaders.append(
                DataLoader(
                    ProsQADataset(data, tokenizer, stage, config.max_length),
                    batch_size=config.batch_size,
                    shuffle=shuffle,
                    collate_fn=BatchCollator(tokenizer.pad_token_id),
                    num_workers=config.num_workers,
                    pin_memory=device.type == "cuda",
                    generator=generator if shuffle else None,
                )
            )
        # Check all lengths before spending compute on this stage.
        for loader in loaders:
            for index in range(len(loader.dataset)):
                loader.dataset[index]
        train_loader, validation_loader = loaders
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        steps = math.ceil(len(train_loader) / config.accumulation_steps) * (epochs - completed)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, min(config.warmup_steps, max(steps - 1, 0)), steps
        )
        for epoch in range(completed + 1, epochs + 1):
            loss, updates = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config.accumulation_steps,
                use_bf16,
            )
            optimizer_steps += updates
            val_loss = validation_loss(model, validation_loader, device, use_bf16)
            record = {
                "stage": stage,
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "train_loss": loss,
                "validation_loss": val_loss,
            }
            history.append(record)
            checkpoint = output / "checkpoints" / f"stage-{stage}_epoch-{epoch}"
            base.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint / "coconut_config.json", {**record, "run": metadata})
            write_json(output / "history.json", history)
            print(json.dumps(record))
    return history
