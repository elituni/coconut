"""Tiny, randomly initialized models and a byte tokenizer; no network downloads."""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from coconut_reproduction.data import SPECIAL_TOKENS


def make_tokenizer(include_answer_marker=True):
    tokens = SPECIAL_TOKENS if include_answer_marker else SPECIAL_TOKENS[:3]
    vocab = {token: i for i, token in enumerate((*tokens, "<eos>", "<unk>"))}
    for character in sorted(pre_tokenizers.ByteLevel.alphabet()):
        if character not in vocab:
            vocab[character] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token="<unk>",
        additional_special_tokens=list(tokens),
    )


@pytest.fixture
def tokenizer():
    return make_tokenizer()


@pytest.fixture
def legacy_tokenizer():
    return make_tokenizer(include_answer_marker=False)


@pytest.fixture
def base_model(tokenizer):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_ctx=256,
            n_embd=16,
            n_layer=2,
            n_head=2,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.eos_token_id,
            attn_implementation="eager",
        )
    )


@pytest.fixture
def examples():
    return [
        {
            "question": "Is Ada a tree?",
            "steps": ["Ada is a maple.", "Maples are trees."],
            "answer": "Yes.",
        },
        {"question": "Is Lee a bird or a tree?", "steps": ["Lee is a bird."], "answer": "Bird."},
        {"question": "Is Jo here?", "steps": [], "answer": "Yes."},
    ]
