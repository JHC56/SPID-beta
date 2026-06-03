# Contributing to SPID

Thanks for your interest in improving SPID! This is a research/portfolio project, but contributions are welcome.

## Ways to Contribute

- **Report bugs** — Open an issue describing what went wrong, with a minimal example.
- **Suggest improvements** — New attack patterns SPID misses, better splitting heuristics, etc.
- **Add training data** — SPID can be fine-tuned on new prompt injection examples (see below).
- **Improve docs** — Fixes to the README, code comments, or examples.

## Reporting a Missed Attack

SPID is known to miss paraphrased and obfuscated attacks. If you find a case it should catch:

1. Open an issue titled `Missed attack: <short description>`
2. Include the exact input text and the score SPID returned
3. Explain why it should have been blocked

## Fine-tuning on Your Own Data

If SPID misses patterns specific to your use case, you can fine-tune it:

```bash
python train.py --additional_data your_attacks.jsonl
```

Your data should be a JSONL file with `text` and `label` fields (0 = safe, 1 = unsafe).

## Code Style

- Keep changes minimal and focused
- Match the existing style in the file you're editing
- Run the existing pipeline to make sure nothing breaks before opening a PR

## Pull Requests

1. Fork the repo and create a branch
2. Make your change
3. Describe what you changed and why
4. Open the PR

## Questions

Open an issue with the `question` label.
