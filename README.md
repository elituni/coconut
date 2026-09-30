# Coconut implementation

My casual implementation of **Coconut (Chain of Continuous Thought)** using GPT-2.

The idea is inspired by [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) by Hao et al. Instead of turning every reasoning step into text, you feed the model's hidden state back in as the next embedding. I tested on ProsQA, a synthetic logical reasoning dataset introduced in the same paper.

The [authors' repo](https://github.com/facebookresearch/coconut) has the official implementation and paper experiment settings. This is my own implementation study based on theoretical concepts introduced in the paper, but with potentially different implementation choices.


## What's here

- A GPT-2 wrapper that feeds hidden states back as continuous thoughts, with batched KV caching.
- A curriculum that gradually replaces written reasoning steps with latent thoughts.
- Training, checkpoint loading, greedy generation, and exact-match evaluation.
- A few configs and CPU tests, including a check that the batched latent loop and its gradients match a simple sequential version.

```text
src/coconut_reproduction/
  data.py         # sequence formatting and padding
  model.py        # the continuous-thought loop
  training.py     # curriculum and checkpoints
  inference.py    # latent reasoning, then text generation
  cli.py          # commands below
configs/          # full run and tiny pipeline check
scripts/          # ProsQA downloader
```


## Results so far

I checked both earlier ProsQA checkpoints on all 300 examples in the supplied validation split:

| Checkpoint | Fixed 6 thoughts | Adaptive, cap 12 |
| --- | --- | --- |
| `cp15` | 89.67% (269/300) | 89.67% (269/300) |
| `cp6-t2` | 88.00% (264/300) | 88.33% (265/300) |

These runs use FP32 on CPU, greedy decoding, a newline after `<eot>`, and plain-answer exact match. [docs/results.json](docs/results.json) has the checkpoint/data hashes and full settings. Model weights and raw prediction files stay local and aren't tracked in Git.

The current four-token notebook's saved output that shows **241/300 (80.33%)** has work-in-progress edits that attempt to improve training speed and performance (but obviously fail); its source and dataset hashes are in [docs/archived_result.json](docs/archived_result.json).

The cleanup fixes an extra-thought issue in generation and restores the newline used during training. It also changes padding, loss weighting, and precision handling, so scores and training trajectories may differ from the notebook. [docs/CHANGES.md](docs/CHANGES.md) has the details.



## Credit

The method and ProsQA data come from [Hao et al.](https://arxiv.org/abs/2412.06769) and the [official Coconut repo](https://github.com/facebookresearch/coconut). GPT-2 runs through [Hugging Face Transformers](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/gpt2).

I used ChatGPT Sol to help clean up and organize the notebook code into this repo. It also helped write documentation, fix generation and training issues, add configs, command-line tools, tests, and GitHub Actions, and run checkpoint evaluations.
