#!/usr/bin/env python3
"""SPID + Gemini Flash — Terminal Demo

Run locally in your terminal:
  python terminal_demo.py

Requires:
  - SPID model (local directory or HuggingFace ID)
  - Gemini API key (https://aistudio.google.com/apikey)
"""

import os
import sys
import json
import time
import re
from getpass import getpass


# ─── Config ────────────────────────────────────────────────────────────
SPID_MODEL_PATH = os.environ.get("SPID_MODEL_PATH", "./spid-deberta-base")


# ─── Dependency check ──────────────────────────────────────────────────
try:
    import torch
    import google.generativeai as genai
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    print(f"Missing dependency: {e.name}")
    print("Install with:")
    print("  pip install torch transformers google-generativeai")
    sys.exit(1)


# ─── Gemini setup ──────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = getpass("Enter Gemini API Key: ").strip()
genai.configure(api_key=api_key)
GEMINI_MODEL = "gemini-2.5-flash-lite"
gemini = genai.GenerativeModel(GEMINI_MODEL)


# ─── Load SPID ─────────────────────────────────────────────────────────
print(f"Loading SPID from {SPID_MODEL_PATH}...")

if not os.path.isdir(SPID_MODEL_PATH):
    # Try as HuggingFace ID
    print(f"(local directory not found, trying HuggingFace Hub)")

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spid_tokenizer = AutoTokenizer.from_pretrained(SPID_MODEL_PATH)
    spid_model = AutoModelForSequenceClassification.from_pretrained(SPID_MODEL_PATH).to(device)
    spid_model.eval()
except Exception as e:
    print(f"Failed to load model: {e}")
    print("\nSet SPID_MODEL_PATH environment variable to your model directory:")
    print("  export SPID_MODEL_PATH=/path/to/spid-deberta-base")
    print("  python terminal_demo.py")
    sys.exit(1)

# Load spid_config.json
if os.path.isdir(SPID_MODEL_PATH):
    cfg_path = f"{SPID_MODEL_PATH}/spid_config.json"
else:
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=SPID_MODEL_PATH, filename="spid_config.json")

with open(cfg_path) as f:
    cfg = json.load(f)
TEMPERATURE = cfg["temperature"]
THRESHOLD = cfg["threshold"]


# ─── SPID functions ────────────────────────────────────────────────────
CONJUNCTIONS = [
    'but', 'however', 'although', 'though', 'while', 'whereas',
    'and', 'or', 'so', 'yet', 'because', 'since', 'as', 'when',
    'if', 'unless', 'until', 'also', 'then', 'first'
]


def split_sentence(text):
    parts = re.split(r'[.;!?,]+', text)
    fragments = []
    for part in parts:
        sub = re.split(
            r'\b(?:' + '|'.join(CONJUNCTIONS) + r')\b',
            part, flags=re.IGNORECASE
        )
        fragments.extend(s.strip() for s in sub if s.strip())
    return [f for f in fragments if len(f) >= 3]


@torch.no_grad()
def classify_spid(text):
    inputs = spid_tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256, padding=True).to(device)
    logits = spid_model(**inputs).logits
    probs = torch.softmax(logits / TEMPERATURE, dim=-1).cpu().numpy()[0]
    return float(probs[1])


def spid_pipeline(text):
    result = {"input": text, "blocked": False, "fragments": []}

    full_prob = classify_spid(text)
    result["full"] = {"prob": full_prob, "blocked": full_prob >= THRESHOLD}
    if result["full"]["blocked"]:
        result["blocked"] = True

    for frag in split_sentence(text):
        prob = classify_spid(frag)
        blocked = prob >= THRESHOLD
        result["fragments"].append({"text": frag, "prob": prob, "blocked": blocked})
        if blocked:
            result["blocked"] = True

    return result


# ─── Gemini function ───────────────────────────────────────────────────
def call_gemini(user_input):
    start = time.time()
    try:
        response = gemini.generate_content(
            user_input,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
            )
        )
        elapsed = time.time() - start
        try:
            tokens = response.usage_metadata.total_token_count
        except:
            tokens = len(user_input.split()) + len(response.text.split())
        return {"success": True, "text": response.text, "tokens": tokens, "time_ms": int(elapsed * 1000)}
    except Exception as e:
        return {"success": False, "error": str(e), "time_ms": int((time.time() - start) * 1000)}


# ─── Main loop ─────────────────────────────────────────────────────────
def process_input(user_input):
    spid_start = time.time()
    spid_result = spid_pipeline(user_input)
    spid_time = int((time.time() - spid_start) * 1000)

    full = spid_result["full"]
    full_tag = "BLOCK" if full["blocked"] else "PASS"

    print(f"  [SPID] {spid_time}ms")
    print(f"  full text    unsafe={full['prob']:.3f}  [{full_tag}]")

    if len(spid_result["fragments"]) > 1:
        for i, f in enumerate(spid_result["fragments"], 1):
            tag = "BLOCK" if f["blocked"] else "PASS"
            preview = f["text"][:50] + ("..." if len(f["text"]) > 50 else "")
            print(f"  fragment {i}   unsafe={f['prob']:.3f}  [{tag}]  {preview}")

    print()

    if spid_result["blocked"]:
        print(f"  [GEMINI] not called  // blocked by SPID")
        print()
        return

    gemini_result = call_gemini(user_input)
    if gemini_result["success"]:
        print(f"  [GEMINI] {gemini_result['time_ms']}ms  tokens={gemini_result['tokens']}")
        for line in gemini_result["text"].split("\n"):
            print(f"  {line}")
    else:
        print(f"  [ERROR] {gemini_result['error']}")
    print()


def main():
    print()
    print(f"  SPID + Gemini Flash  |  type 'exit' to quit, 'clear' to reset")
    print()

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print()
            print(f"  SPID + Gemini Flash  |  type 'exit' to quit, 'clear' to reset")
            print()
            continue

        print()
        process_input(user_input)


if __name__ == "__main__":
    main()
