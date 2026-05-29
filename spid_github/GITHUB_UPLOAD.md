# How to Upload SPID to GitHub (Public)

Step-by-step guide to publish this project as a public GitHub repository.

## Before You Start

Replace `your-username` placeholders in these files with your actual usernames:

- `README.md` — replace `your-username` (HuggingFace) and `your-org` (GitHub)
- `MODEL_CARD.md` — replace `your-username`
- Add your real videos to `demo/` and the architecture image to `img/`

## Step 1: Add Your Media Files

```
demo/
├── spid_blocks.mp4                  (rename Video_Project_4.mp4)
└── spid_missed_gemini_caught.mp4    (rename Video_Project_5.mp4)

img/
└── spid_architecture.png            (your diagram)
```

Then delete the placeholder `demo/README.md` and `img/README.md`.

## Step 2: Create the Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `spid` (or `SPID`)
3. Description: `Split-based Prompt Injection Detector — lightweight pre-filter for LLM APIs`
4. Select **Public**
5. Do NOT check "Add a README" (you already have one)
6. Click **Create repository**

## Step 3: Upload from Your Computer

### Option A: Command line (recommended)

Open a terminal in your project folder:

```bash
cd path/to/spid_github

git init
git add .
git commit -m "Initial commit: SPID prompt injection detector"
git branch -M main
git remote add origin https://github.com/your-username/spid.git
git push -u origin main
```

If prompted for credentials, use a Personal Access Token (not your password):
https://github.com/settings/tokens

### Option B: Web upload (no git needed)

1. On the new repo page, click **uploading an existing file**
2. Drag and drop all files and folders
3. Write a commit message
4. Click **Commit changes**

Note: GitHub web upload has a 25MB per-file limit. Your videos (4.2MB, 1.5MB) are fine.

## Step 4: Verify

Visit `https://github.com/your-username/spid` and check:

- README renders with the architecture image
- Videos play (click on them)
- Code files are present

## Notes

- The model files (`spid-deberta-base/`) are excluded by `.gitignore`. Upload those to HuggingFace instead (see `HUGGINGFACE_UPLOAD.md`).
- GitHub renders `.mp4` videos inline in the README automatically.
- If videos don't autoplay, GitHub shows them as a player — that's normal.

## File Checklist

```
Code:
  config.py, utils.py, pipeline.py, train.py,
  evaluate.py, demo.py, run.py, terminal_demo.py

Notebook:
  notebooks/spid_final.ipynb

Docs:
  README.md, LICENSE, requirements.txt
  MODEL_CARD.md, HUGGINGFACE_UPLOAD.md

Media (you add):
  demo/spid_blocks.mp4
  demo/spid_missed_gemini_caught.mp4
  img/spid_architecture.png

Config:
  .gitignore
```
