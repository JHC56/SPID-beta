"""SPID — inference pipeline: SPID only (no ensemble)."""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score

from config import DEVICE, MAX_LEN, MODEL_NAME
from utils import split_sentence


# ═══════════════════════════════════════════════════════════════════
#  SPID Classifier
# ═══════════════════════════════════════════════════════════════════

class SPIDPipeline:
    """SPID: Split-based Prompt Injection Detector.
    Splits text into fragments and classifies each independently."""

    def __init__(self, spid_model, spid_tokenizer, temperature, threshold):
        self.spid_model = spid_model
        self.spid_tokenizer = spid_tokenizer
        self.temperature = temperature
        self.threshold = threshold

    @classmethod
    def from_pretrained(cls, spid_dir: str, device=None):
        """Load SPID from local directory or HuggingFace Hub.
        
        Args:
            spid_dir: Local path (e.g., './spid-deberta-base') OR 
                     HuggingFace Hub ID (e.g., 'username/spid-deberta-base')
        """
        import os
        device = device or DEVICE

        spid_tokenizer = AutoTokenizer.from_pretrained(spid_dir)
        spid_model = AutoModelForSequenceClassification.from_pretrained(
            spid_dir
        ).to(device)
        spid_model.eval()

        # Load spid_config.json (temperature + threshold)
        if os.path.isdir(spid_dir):
            # Local directory
            cfg_path = f"{spid_dir}/spid_config.json"
        else:
            # HuggingFace Hub
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_id=spid_dir, filename="spid_config.json")
        
        with open(cfg_path) as f:
            cfg = json.load(f)
        temperature = cfg["temperature"]
        threshold = cfg["threshold"]

        print(f"SPID loaded from {spid_dir} (T={temperature}, thr={threshold})")

        return cls(spid_model, spid_tokenizer, temperature, threshold)

    @torch.no_grad()
    def classify(self, text: str) -> np.ndarray:
        """Classify text, return [safe_prob, unsafe_prob]."""
        inputs = self.spid_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LEN, padding=True,
        ).to(self.spid_model.device)
        logits = self.spid_model(**inputs).logits
        probs = torch.softmax(
            logits / self.temperature, dim=-1
        ).cpu().numpy()[0]
        return probs

    def __call__(self, text: str, verbose: bool = True) -> dict:
        """Classify text. Return {'blocked': bool, 'fragments': [...]}.
        
        BLOCK when:
          - Full text unsafe, OR
          - Any fragment unsafe
        """
        result = {"input": text, "blocked": False, "fragments": []}

        # 1) Full-text classification
        spid_full = self.classify(text)
        full_blocked = spid_full[1] >= self.threshold
        if full_blocked:
            result["blocked"] = True
        result["full"] = {
            "unsafe_prob": float(spid_full[1]),
            "blocked": full_blocked,
        }

        # 2) Fragment classification (splitting)
        fragments = split_sentence(text)
        for frag in fragments:
            if len(frag.strip()) < 3:
                continue
            spid_frag = self.classify(frag)
            frag_blocked = spid_frag[1] >= self.threshold
            if frag_blocked:
                result["blocked"] = True
            result["fragments"].append({
                "text": frag,
                "unsafe_prob": float(spid_frag[1]),
                "blocked": frag_blocked,
            })

        if verbose:
            print(f"  input: {text[:80]}{'...' if len(text) > 80 else ''}")
            ftag = "BLOCKED" if result["full"]["blocked"] else "pass"
            print(f"    [full]  unsafe={result['full']['unsafe_prob']:.2f}  -> {ftag}")
            for fr in result["fragments"]:
                ftag2 = "BLOCKED" if fr["blocked"] else "pass"
                print(f"    [frag]  unsafe={fr['unsafe_prob']:.2f}  -> {ftag2}  {fr['text'][:50]}")
            print(f"  => {'BLOCKED' if result['blocked'] else 'PASSED'}")

        return result


# ═══════════════════════════════════════════════════════════════════
#  OOD pipeline evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_pipeline_ood(pipe, ood_data, ood_labels, spid_preds_final):
    """Run full pipeline (splitting) on OOD data."""
    print(f"=== OOD pipeline (threshold={pipe.threshold}, "
          f"T={pipe.temperature:.1f}) ===")

    pipe_preds = []
    for x in ood_data:
        r = pipe(x["text"], verbose=False)
        pipe_preds.append(1 if r["blocked"] else 0)
    pipe_preds = np.array(pipe_preds)

    print(f"\nclassifier only (full text):")
    print(f"  precision: {precision_score(ood_labels, spid_preds_final):.4f}")
    print(f"  recall:    {recall_score(ood_labels, spid_preds_final):.4f}")
    print(f"  f1:        {f1_score(ood_labels, spid_preds_final):.4f}")

    print(f"\nfull pipeline (split + classify):")
    print(f"  precision: {precision_score(ood_labels, pipe_preds):.4f}")
    print(f"  recall:    {recall_score(ood_labels, pipe_preds):.4f}")
    print(f"  f1:        {f1_score(ood_labels, pipe_preds):.4f}")
