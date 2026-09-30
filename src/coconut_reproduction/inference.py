"""Greedy latent reasoning and exact-match evaluation without ground-truth steps."""

from dataclasses import asdict, dataclass

import torch
from tqdm.auto import tqdm

from .data import SPECIAL_TOKENS, token_ids


@dataclass
class Generation:
    text: str
    num_thoughts: int
    stopped_on_eot: bool
    ended_on_eos: bool


def extract_answer(text: str, eos_token: str, answer_format: str = "marker") -> str:
    """Require the trained answer separator; normalize case and outer whitespace."""
    if answer_format == "plain":
        return text.replace(eos_token, "").strip().lower()
    if answer_format != "marker":
        raise ValueError("answer_format must be marker or plain.")
    if "<ans>" not in text:
        return ""
    return text.rsplit("<ans>", 1)[1].replace(eos_token, "").strip().lower()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    question: str,
    *,
    max_thoughts: int = 12,
    max_new_tokens: int = 50,
    latent_policy: str = "adaptive",
    eot_newline: bool = True,
) -> Generation:
    """Feed hidden states directly until <eot> is predicted or a fixed cap is reached.

    The count is the number of latent embeddings actually consumed. A prediction
    of <eot> at <bot> permits zero thoughts. Fixed mode always consumes the cap.
    Text starts after the same <eot> + newline separator used in training.
    """
    if max_thoughts < 0 or max_new_tokens < 1:
        raise ValueError("max_thoughts must be nonnegative and max_new_tokens positive.")
    if latent_policy not in ("adaptive", "fixed"):
        raise ValueError("latent_policy must be adaptive or fixed.")
    if not question.strip() or any(token in question for token in SPECIAL_TOKENS):
        raise ValueError("Question must be nonempty and contain no reserved control tokens.")
    ids = token_ids(tokenizer, require_answer_marker=False)
    device = next(model.parameters()).device
    prefix = tokenizer.encode(f"{question}\n<bot>", add_special_tokens=False)
    suffix = [ids["<eot>"]]
    if eot_newline:
        suffix.extend(tokenizer.encode("\n", add_special_tokens=False))
    if len(prefix) + max_thoughts + len(suffix) + max_new_tokens > model.config.n_positions:
        raise ValueError("Question and generation budget exceed the GPT-2 context window.")
    model.eval()
    outputs = model(
        input_ids=torch.tensor([prefix], device=device), output_hidden_states=True, use_cache=True
    )
    count = 0
    stopped = False
    while count < max_thoughts:
        if latent_policy == "adaptive" and int(outputs.logits[0, -1].argmax()) == ids["<eot>"]:
            stopped = True
            break
        outputs = model(
            inputs_embeds=outputs.hidden_states[-1][:, -1:, :],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        count += 1
    if latent_policy == "adaptive" and int(outputs.logits[0, -1].argmax()) == ids["<eot>"]:
        stopped = True
    outputs = model(
        input_ids=torch.tensor([suffix], device=device),
        past_key_values=outputs.past_key_values,
        use_cache=True,
    )
    generated = []
    for _ in range(max_new_tokens):
        next_id = int(outputs.logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        outputs = model(
            input_ids=torch.tensor([[next_id]], device=device),
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
    return Generation(
        text=tokenizer.decode(generated, skip_special_tokens=False),
        num_thoughts=count,
        stopped_on_eot=stopped,
        ended_on_eos=generated[-1] == tokenizer.eos_token_id,
    )


def evaluate(
    model, tokenizer, examples, *, answer_format: str = "marker", **generation_options
) -> dict:
    """Evaluate all supplied examples and retain predictions for auditability."""
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset.")
    predictions = []
    correct = 0
    for index, item in enumerate(tqdm(examples, desc="Exact match")):
        result = generate(model, tokenizer, item["question"], **generation_options)
        answer = extract_answer(result.text, tokenizer.eos_token, answer_format)
        match = answer == item["answer"].strip().lower()
        correct += int(match)
        predictions.append(
            {
                "index": index,
                "expected": item["answer"],
                "answer": answer,
                "correct": match,
                **asdict(result),
            }
        )
    return {
        "correct": correct,
        "total": len(examples),
        "accuracy": correct / len(examples),
        "generation": generation_options,
        "answer_format": answer_format,
        "predictions": predictions,
    }
