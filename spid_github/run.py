"""SPID — full workflow: train → evaluate → demo.

Usage:
    python run.py              # full pipeline (train + eval + demo)
    python run.py --skip-train # load saved model, run eval + demo
"""

import argparse

from config import DEVICE, MODEL_NAME


def main(skip_train: bool = False):
    print(f"device: {DEVICE}")
    print(f"model: {MODEL_NAME}")

    if not skip_train:
        # ── Step 1: Train ──
        from train import train
        model, tokenizer, trainer, train_list, test_list, \
            ood_data, ood_labels, ood_tok, test_tok = train()

        # ── Step 2: Calibrate ──
        from evaluate import calibrate_temperature, indist_sweep, evaluate_ood
        temperature, test_probs_cal = calibrate_temperature(
            trainer, test_tok, test_list
        )
        indist_sweep(test_probs_cal, test_list)

        # ── Step 3: OOD Evaluation ──
        threshold, temperature, ood_probs_cal, spid_preds_final = evaluate_ood(
            trainer, ood_tok, ood_data, ood_labels, temperature
        )

        # ── Step 4: Build pipeline ──
        from pipeline import SPIDPipeline, evaluate_pipeline_ood
        pipe = SPIDPipeline(model, tokenizer, temperature, threshold)

        # ── Step 5: Pipeline OOD eval ──
        evaluate_pipeline_ood(pipe, ood_data, ood_labels, spid_preds_final)

    else:
        # Load saved model
        from pipeline import SPIDPipeline
        pipe = SPIDPipeline.from_pretrained("./spid-deberta-base")

    # ── Step 6: Demo ──
    from demo import run_demo, splitting_comparison
    run_demo(pipe)
    splitting_comparison(pipe)

    print("\n=== All steps complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPID full workflow")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training, load saved model from ./spid-deberta-base",
    )
    args = parser.parse_args()
    main(skip_train=args.skip_train)
