"""
evaluator.py

Responsibilities:
- Load scores of all 3 trained models (lr, rf, svm)
- Display comparison table
- Identify best model by F1 score
- Save best model as best_model.pkl

Usage:
    evaluator = Evaluator()
    evaluator.evaluate()
"""

import json
import shutil
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR      = Path("models/saved")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

MODEL_KEYS = ["lr", "rf", "svm"]


# ── Evaluator Class ───────────────────────────────────────────────────────────

class Evaluator:

    # -------------------------------------------------------------------------
    # SECTION 1: LOAD SCORES
    # -------------------------------------------------------------------------

    def _load_scores(self) -> list:
        """
        Loads score JSON files for all trained models.

        Returns:
            list: List of score dicts for each model found

        Raises:
            FileNotFoundError : If no score files found at all
        """
        scores = []

        for key in MODEL_KEYS:
            path = MODELS_DIR / f"{key}_scores.json"

            if not path.exists():
                logger.warning(f"{key}_scores.json not found — skipping {key}")
                continue

            try:
                with open(path, "r") as f:
                    data = json.load(f)
                scores.append(data)
                logger.info(f"Loaded scores for: {data['model_name']}")

            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

        if not scores:
            raise FileNotFoundError(
                "No score files found in models/saved/. "
                "Run trainer.py first."
            )

        return scores

    # -------------------------------------------------------------------------
    # SECTION 2: DISPLAY COMPARISON TABLE
    # -------------------------------------------------------------------------

    def _display_table(self, scores: list):
        """
        Displays a comparison table of all trained models.

        Args:
            scores (list): List of score dicts
        """
        header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
        divider = "─" * 70

        print(f"\n{divider}")
        print(f"  Model Comparison")
        print(f"{divider}")
        print(header)
        print(divider)

        for s in scores:
            row = (
                f"{s['model_name']:<25} "
                f"{s['accuracy']*100:>9.2f}% "
                f"{s['precision']*100:>9.2f}% "
                f"{s['recall']*100:>9.2f}% "
                f"{s['f1']*100:>9.2f}%"
            )
            print(row)

        print(divider)

        logger.info("Model comparison table displayed.")

    # -------------------------------------------------------------------------
    # SECTION 3: FIND BEST MODEL
    # -------------------------------------------------------------------------

    def _find_best(self, scores: list) -> dict:
        """
        Finds best model by highest F1 score.

        Args:
            scores (list): List of score dicts

        Returns:
            dict: Score dict of the best model
        """
        best = max(scores, key=lambda x: x["f1"])
        logger.info(f"Best model by F1: {best['model_name']} ({best['f1']*100:.2f}%)")
        return best

    # -------------------------------------------------------------------------
    # SECTION 4: SAVE BEST MODEL
    # -------------------------------------------------------------------------

    def _save_best_model(self, best: dict):
        """
        Copies best model .pkl to best_model.pkl

        Args:
            best (dict): Score dict of best model

        Raises:
            FileNotFoundError : If best model .pkl not found
            RuntimeError      : If saving fails
        """
        src = MODELS_DIR / f"{best['model_key']}_model.pkl"

        if not src.exists():
            raise FileNotFoundError(
                f"{best['model_key']}_model.pkl not found. "
                f"Run trainer.train('{best['model_key']}') first."
            )

        try:
            shutil.copy(src, BEST_MODEL_PATH)
            logger.info(f"Best model saved to {BEST_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to save best model: {e}")

    # -------------------------------------------------------------------------
    # SECTION 5: MAIN EVALUATE FUNCTION
    # -------------------------------------------------------------------------

    def evaluate(self):
        """
        Main function.
        Loads scores → displays table → finds best → saves best_model.pkl

        Raises:
            FileNotFoundError : If no score files found or model .pkl missing
            RuntimeError      : If saving best model fails
        """
        logger.info("=" * 55)
        logger.info("Starting evaluation...")
        logger.info("=" * 55)

        # Step 1: Load scores
        scores = self._load_scores()

        # Step 2: Display comparison table
        self._display_table(scores)

        # Step 3: Find best model
        best = self._find_best(scores)

        print(f"\n  Best Model : {best['model_name']}")
        print(f"  F1 Score   : {best['f1']*100:.2f}%")

        # Step 4: Save best model
        self._save_best_model(best)

        print(f"\n✅ best_model.pkl saved → {BEST_MODEL_PATH}")

        logger.info("=" * 55)
        logger.info("Evaluation complete.")
        logger.info(f"  Best model : {best['model_name']}")
        logger.info(f"  F1 Score   : {best['f1']*100:.2f}%")
        logger.info("=" * 55)


# ── CLI / Quick Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluator = Evaluator()

    try:
        evaluator.evaluate()

    except FileNotFoundError as e:
        print(f"\n File not found: {e}")

    except RuntimeError as e:
        print(f"\n Runtime error: {e}")

    except Exception as e:
        print(f"\n Unexpected error: {e}")