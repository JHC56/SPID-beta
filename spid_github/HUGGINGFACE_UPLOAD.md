# Uploading SPID to Hugging Face Hub

Step-by-step guide to publish your trained model so others can load it with one line.

## Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login    # paste your token from https://huggingface.co/settings/tokens
```

## Step 1: Prepare the Model Folder

Your trained model folder should contain:

```
spid-deberta-base/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── spm.model
├── special_tokens_map.json
├── added_tokens.json
└── spid_config.json        # temperature + threshold (important!)
```

If your folder is named differently (e.g. `spid-deberta-large`), that's fine—just point to it in the next steps.

## Step 2: Add the Model Card

Copy the model card into the folder as `README.md`:

```bash
cp MODEL_CARD.md spid-deberta-base/README.md
```

Then open `spid-deberta-base/README.md` and replace every `your-username` with your actual Hugging Face username, and `your-org` with your GitHub username.

## Step 3: Upload

### Option A — Command line

```bash
huggingface-cli upload your-username/spid-deberta-base ./spid-deberta-base
```

This creates the repo automatically if it doesn't exist.

### Option B — Python

```python
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="your-username/spid-deberta-base", exist_ok=True)
api.upload_folder(
    folder_path="./spid-deberta-base",
    repo_id="your-username/spid-deberta-base",
    repo_type="model",
)
```

## Step 4: Verify

Visit `https://huggingface.co/your-username/spid-deberta-base`

You should see:
- The model card with badges and metrics
- A "Use this model" button
- Tags (prompt-injection, llm-security, etc.) in the header
- Performance metrics auto-rendered from the `model-index` metadata

Test loading:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("your-username/spid-deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("your-username/spid-deberta-base")
print("Loaded successfully")
```

## Step 5: Link from GitHub

In your GitHub `README.md`, the model badge and links now point to your Hub page. Add a badge near the top if you like:

```markdown
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/your-username/spid-deberta-base)
```

## What Gets Uploaded

| File | Purpose |
|:-----|:--------|
| `config.json` | Model architecture config |
| `model.safetensors` | Weights (~700MB) |
| `tokenizer.*`, `spm.model` | Tokenizer |
| `spid_config.json` | SPID temperature + threshold |
| `README.md` | Model card (rendered on the Hub page) |

That's it. Anyone can now load your model with a single line.
