# Changes from the supplied notebook

## Structure and usability

- Replaced the notebook's interdependent cells with a Python package: data formatting, model, inference, training, and CLI modules.
- Replaced inline installs, unused imports, duplicated checkpoint-loading/evaluation code, and hardcoded restart instructions with package dependencies, configs, and documented commands.
- Added a synthetic sample dataset, pinned upstream data downloader, source/hash manifests, paper citation, and third-party notices.
- Added CPU tests and a GitHub Actions workflow, with no pretrained model downloads in tests.
- The original `coconut.zip` remains outside this repository and is unchanged. No original notebook output, installation notebook, or token file is copied into the publication files.

## Preserved research approach

- GPT-2; one continuous thought per replaced reasoning step.
- Per-example latent count `min(stage, len(steps))`.
- Cached batched latent recurrence followed by embedding injection and a full language-model pass.
- Question/latent masking, including supervision of `<eot>` and `<ans>`.
- Default stages 0–6, 50 total epochs, batch size 32, learning rate `1e-4`, weight decay `0.01`, and stage-level optimizer resets.

## Correctness and behavior changes

- Count latent embeddings actually fed into the model. The original generator initialized a thought list before its loop and could assemble up to 13 thoughts with a loop bound of 12.
- Append `<eot>\n` before text generation, matching the training separator. The original latent generator appended only `<eot>`.
- Use the KV cache directly during greedy inference rather than relying on `generate(inputs_embeds=...)` return conventions.
- Make zero-thought generation explicit, and infer it for recorded stage-0 checkpoints.
- Require `<ans>` for answer extraction by default. The notebook also accepted an `<eot>` fallback. Older three-token checkpoints can explicitly use `--answer-format plain` to score the complete generated continuation, and `--no-eot-newline` to explore their separator convention.
- Dynamically pad batches and preserve real EOS targets even when EOS is also the padding token.
- Reject overlong training examples and inference budgets; the notebook silently truncated training sequences.
- Normalize accumulated gradients by the group's total supervised next-token count. This handles partial groups and unequal target lengths. The notebook divided every batch by a fixed accumulation count, including a partial final group.
- Count scheduler updates with a ceiling, including partial accumulation groups.
- Compute validation loss weighted by target tokens, rather than averaging batch means.
- Seed training and data shuffling; use ordered validation and full-split standalone evaluation instead of random 100-example progress evaluations.
- Save run settings, environment versions, dataset hashes, checkpoint stages, losses, and individual evaluation predictions. Refuse to overwrite existing run/evaluation output.
- Continue from explicit stage/epoch settings with model weights only; do not imply exact optimizer/RNG restoration.
- Load weights in FP32 and optionally use BF16 autocast on supported CUDA devices. The notebook loaded the underlying weights in BF16. Use eager attention for a consistent, tested GPT-2 path; the notebook used SDPA during training.
- Replace interactive plots with machine-readable loss history. A plot can be generated from `history.json` without a plotting dependency in the training path.

These changes may affect training trajectories and checkpoint accuracy. Archived notebook results are evidence of the original run and are not results of the revised implementation.

## Older checkpoints

The two supplied older checkpoints are for ProsQA and use a three-token format without `<ans>`. Their original notebook versions weren't saved. The CLI supports them with an explicit plain-answer option, and records the generation settings with each evaluation. No ProntoQA checkpoint or data is included.
