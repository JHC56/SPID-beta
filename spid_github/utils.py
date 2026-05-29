"""SPID — utilities: splitting algorithm and data loading."""

import re
import random
import numpy as np
import pandas as pd
from datasets import load_dataset

from config import SEED

# ═══════════════════════════════════════════════════════════════════
#  Splitting algorithm  (Cell 9)
# ═══════════════════════════════════════════════════════════════════

EN_CONJUNCTIONS = [
    "but", "however", "and then", "also", "so", "because",
    "therefore", "meanwhile", "furthermore", "additionally",
    "nevertheless", "moreover", "yet", "still", "instead",
    "otherwise", "hence", "thus", "although", "though",
]


def split_sentence(text: str) -> list[str]:
    """Split text into fragments on punctuation and conjunctions."""
    chunks = re.split(r'[.;!?]+', text)
    expanded = []
    for chunk in chunks:
        expanded.extend(re.split(r',', chunk))
    final = []
    for chunk in expanded:
        parts = [chunk]
        for conj in sorted(EN_CONJUNCTIONS, key=len, reverse=True):
            new_parts = []
            for p in parts:
                splits = re.split(
                    r'\b' + re.escape(conj) + r'\b', p, flags=re.IGNORECASE
                )
                new_parts.extend(splits)
            parts = new_parts
        final.extend(parts)
    result = [s.strip() for s in final if len(s.strip()) >= 3]
    return result if result else [text.strip()]


# ═══════════════════════════════════════════════════════════════════
#  Data helpers  (Cells 4-6)
# ═══════════════════════════════════════════════════════════════════

def _extract_human(text: str):
    """Extract human turn from Anthropic hh-rlhf format."""
    if "Human:" in text and "Assistant:" in text:
        try:
            return text.split("Human:")[1].split("Assistant:")[0].strip()
        except Exception:
            return None
    return None


def load_attack_data():
    """Load attack datasets: AdvBench, deepset, Gandalf."""
    # AdvBench
    adv_df = pd.read_csv(
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
        "main/data/advbench/harmful_behaviors.csv"
    )
    adv_attacks = adv_df["goal"].tolist()

    # deepset
    deepset = load_dataset("deepset/prompt-injections", split="train")
    deepset_attacks = [x["text"] for x in deepset if x["label"] == 1]
    deepset_benign = [x["text"] for x in deepset if x["label"] == 0]

    # Gandalf
    gandalf = load_dataset(
        "Lakera/gandalf_ignore_instructions", split="train"
    )
    gandalf_attacks = [x["text"] for x in gandalf]

    print(f"attacks: advbench {len(adv_attacks)}, "
          f"deepset {len(deepset_attacks)}, gandalf {len(gandalf_attacks)}")
    print(f"deepset benign: {len(deepset_benign)}")

    return adv_attacks, deepset_attacks, deepset_benign, gandalf_attacks


def load_benign_data():
    """Load benign datasets: hh-rlhf, Dolly, OpenAssistant."""
    # hh-rlhf
    hh = load_dataset(
        "Anthropic/hh-rlhf", data_dir="helpful-base", split="train"
    )
    hh_normals = []
    for x in hh.shuffle(seed=SEED).select(range(3000)):
        h = _extract_human(x["chosen"])
        if h and 5 < len(h) < 1500:
            hh_normals.append(h)
        if len(hh_normals) >= 1000:
            break

    # Dolly
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_normals = [
        x["instruction"] for x in dolly.shuffle(seed=SEED)
        if 5 < len(x["instruction"]) < 1500
    ][:1000]

    # OpenAssistant
    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    oasst_normals = []
    for x in oasst.shuffle(seed=SEED):
        if x.get("role") == "prompter" and x.get("lang") == "en":
            t = x.get("text", "")
            if 5 < len(t) < 1500:
                oasst_normals.append(t)
        if len(oasst_normals) >= 2000:
            break

    print(f"benign: hh-rlhf {len(hh_normals)}, dolly {len(dolly_normals)}, "
          f"oasst {len(oasst_normals)}")

    return hh_normals, dolly_normals, oasst_normals


def load_jailbreakhub_may():
    """Load JailbreakHub May split for augmentation + HNM candidates."""
    print("=== Domain augmentation + HNM candidate prep (JailbreakHub May) ===")
    try:
        jb_attack_may = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "jailbreak_2023_05_07", split="train"
        )
        jb_normal_may = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "regular_2023_05_07", split="train"
        )

        aug_attacks_all = []
        for x in jb_attack_may.shuffle(seed=SEED):
            p = x.get("prompt")
            if isinstance(p, str) and 5 < len(p) < 1500:
                aug_attacks_all.append(p)
            if len(aug_attacks_all) >= 700:
                break

        aug_normals_all = []
        for x in jb_normal_may.shuffle(seed=SEED):
            p = x.get("prompt")
            if isinstance(p, str) and 5 < len(p) < 1500:
                aug_normals_all.append(p)
            if len(aug_normals_all) >= 700:
                break

        # first 500 → training, next 200 → HNM candidates
        aug_attacks = aug_attacks_all[:500]
        aug_normals = aug_normals_all[:500]
        hnm_candidate_attacks = aug_attacks_all[500:700]
        hnm_candidate_normals = aug_normals_all[500:700]

        print(f"  training aug:    {len(aug_attacks)} attacks "
              f"+ {len(aug_normals)} normals")
        print(f"  HNM candidates:  {len(hnm_candidate_attacks)} attacks "
              f"+ {len(hnm_candidate_normals)} normals")

    except Exception as e:
        print(f"  warning: May split load failed: {e}")
        aug_attacks, aug_normals = [], []
        hnm_candidate_attacks, hnm_candidate_normals = [], []

    return aug_attacks, aug_normals, hnm_candidate_attacks, hnm_candidate_normals


def build_dataset():
    """Build full train/test split. Returns train_list, test_list, all data counts,
    and HNM candidates."""
    adv_attacks, deepset_attacks, deepset_benign, gandalf_attacks = load_attack_data()
    hh_normals, dolly_normals, oasst_normals = load_benign_data()
    aug_attacks, aug_normals, hnm_cand_atk, hnm_cand_nrm = load_jailbreakhub_may()

    all_attacks = (adv_attacks[:400] + deepset_attacks[:300]
                   + gandalf_attacks[:350] + aug_attacks)
    all_normals = (hh_normals[:1000] + dolly_normals[:1000]
                   + oasst_normals[:2000] + deepset_benign + aug_normals)

    data = ([{"text": t, "label": 1} for t in all_attacks]
            + [{"text": t, "label": 0} for t in all_normals])
    random.shuffle(data)

    n = len(data)
    n_train = int(n * 0.75)
    train_list = data[:n_train]
    test_list = data[n_train:]

    n_attack = sum(x["label"] for x in data)
    n_benign = n - n_attack
    ratio = n_benign / n_attack

    print(f"total: {n} (attack {n_attack}, benign {n_benign}, ratio 1:{ratio:.1f})")
    print(f"train: {len(train_list)}")
    print(f"test:  {len(test_list)}")

    return train_list, test_list, hnm_cand_atk, hnm_cand_nrm, all_attacks, all_normals


def load_ood_data():
    """Load OOD eval set: JailbreakHub December split."""
    jb_attack = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "jailbreak_2023_12_25", split="train"
    )
    jb_normal = load_dataset(
        "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "regular_2023_12_25", split="train"
    )

    ood_data = []
    cnt = 0
    for x in jb_attack.shuffle(seed=SEED):
        p = x.get("prompt")
        if isinstance(p, str) and 5 < len(p) < 1500:
            ood_data.append({"text": p, "label": 1})
            cnt += 1
        if cnt >= 500:
            break

    cnt = 0
    for x in jb_normal.shuffle(seed=SEED):
        p = x.get("prompt")
        if isinstance(p, str) and 5 < len(p) < 1500:
            ood_data.append({"text": p, "label": 0})
            cnt += 1
        if cnt >= 500:
            break

    ood_labels = np.array([x["label"] for x in ood_data])
    print(f"OOD: {len(ood_data)} (attack {(ood_labels==1).sum()}, "
          f"benign {(ood_labels==0).sum()})")

    return ood_data, ood_labels
