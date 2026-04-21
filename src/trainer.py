# trainer.py
# Trains multiple ML models on the processed dataset.
# Models: Logistic Regression, Random Forest, SVM.
"""
trainer.py

Responsibilities:
- Receive TF-IDF matrices + labels from feature_extractor.py
- Train one model at a time (lr / rf / svm)
- Show Accuracy, Precision, Recall, F1 after training
- Save each model separately as .pkl file
- Save each model scores separately as .json file

Supported models:
    "lr"  → Logistic Regression
    "rf"  → Random Forest
    "svm" → Support Vector Machine

Usage:
    trainer = Trainer(X_train_tfidf, X_test_tfidf, y_train, y_test)
    trainer.train("lr")    # train logistic regression
    trainer.train("rf")    # train random forest
    trainer.train("svm")   # train SVM

Files saved per model:
    models/saved/lr_model.pkl    + models/saved/lr_scores.json
    models/saved/rf_model.pkl    + models/saved/rf_scores.json
    models/saved/svm_model.pkl   + models/saved/svm_scores.json
"""

import json
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from config.config import RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models/saved")

# ── Model Definitions ─────────────────────────────────────────────────────────
MODEL_MAP = {
    "lr": {
        "name": "Logistic Regression",
        "model": LogisticRegression(
            max_iter    = 1000,
            random_state= RANDOM_STATE,
            multi_class = "multinomial",
            solver      = "lbfgs",
        ),
    },
    "rf": {
        "name": "Random Forest",
        "model": RandomForestClassifier(
            n_estimators = 200,
            random_state = RANDOM_STATE,
            n_jobs       = -1,          # use all CPU cores
        ),
    },
    "svm": {
        "name": "Support Vector Machine",
        "model": SVC(
            kernel       = "linear",
            random_state = RANDOM_STATE,
            probability  = True,        # needed for predict_proba in predictor.py
        ),
    },
}


# ── Trainer Class ─────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize trainer with output from feature_extractor.py.

        Args:
            X_train : TF-IDF sparse matrix for training
            X_test  : TF-IDF sparse matrix for testing
            y_train : Category labels for training
            y_test  : Category labels for testing
        """
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        # Create models directory if not exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # SECTION 1: GET MODEL
    # -------------------------------------------------------------------------

    def _get_model(self, model_key: str):
        """
        Returns model name and object for given key.

        Args:
            model_key (str): "lr", "rf", or "svm"

        Returns:
            tuple: (model_name, model_object)

        Raises:
            ValueError: If model_key is not supported
        """
        model_key = model_key.lower().strip()

        if model_key not in MODEL_MAP:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Choose from: {list(MODEL_MAP.keys())}"
            )

        entry = MODEL_MAP[model_key]
        return entry["name"], entry["model"]

    # -------------------------------------------------------------------------
    # SECTION 2: COMPUTE SCORES
    # -------------------------------------------------------------------------

    def _compute_scores(self, y_pred) -> dict:
        """
        Computes Accuracy, Precision, Recall, F1.
        Uses weighted average — correct for imbalanced multi-class data.

        Args:
            y_pred: Predicted labels from model

        Returns:
            dict: {accuracy, precision, recall, f1}
        """
        return {
            "accuracy" : round(accuracy_score(self.y_test, y_pred), 4),
            "precision": round(precision_score(self.y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall"   : round(recall_score(self.y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1"       : round(f1_score(self.y_test, y_pred, average="weighted", zero_division=0), 4),
        }

    # -------------------------------------------------------------------------
    # SECTION 3: SHOW SCORES
    # -------------------------------------------------------------------------

    def _show_scores(self, model_name: str, scores: dict, y_pred):
        """
        Prints scores in clean readable format.
        Shows overall scores + per category breakdown.

        Args:
            model_name : Name of the model
            scores     : Dict of accuracy, precision, recall, f1
            y_pred     : Predicted labels (for classification report)
        """
        logger.info("=" * 55)
        logger.info(f"Results — {model_name}")
        logger.info("=" * 55)
        logger.info(f"  Accuracy  : {scores['accuracy']  * 100:.2f}%")
        logger.info(f"  Precision : {scores['precision'] * 100:.2f}%")
        logger.info(f"  Recall    : {scores['recall']    * 100:.2f}%")
        logger.info(f"  F1 Score  : {scores['f1']        * 100:.2f}%")
        logger.info("=" * 55)
        logger.info("Per-category breakdown:")
        logger.info(
            "\n" + classification_report(
                self.y_test,
                y_pred,
                zero_division=0,
            )
        )

        # Print to console for quick visibility
        print(f"\n{'=' * 55}")
        print(f"Results — {model_name}")
        print(f"{'=' * 55}")
        print(f"  Accuracy  : {scores['accuracy']  * 100:.2f}%")
        print(f"  Precision : {scores['precision'] * 100:.2f}%")
        print(f"  Recall    : {scores['recall']    * 100:.2f}%")
        print(f"  F1 Score  : {scores['f1']        * 100:.2f}%")
        print(f"{'=' * 55}")

    # -------------------------------------------------------------------------
    # SECTION 4: SAVE MODEL
    # -------------------------------------------------------------------------

    def _save_model(self, model, model_key: str):
        """
        Saves trained model as <model_key>_model.pkl

        Args:
            model     : Trained model object
            model_key : "lr", "rf", or "svm"

        Raises:
            RuntimeError: If saving fails
        """
        path = MODELS_DIR / f"{model_key}_model.pkl"
        try:
            joblib.dump(model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")

    # -------------------------------------------------------------------------
    # SECTION 5: SAVE SCORES
    # -------------------------------------------------------------------------

    def _save_scores(self, scores: dict, model_name: str, model_key: str):
        """
        Saves model scores as <model_key>_scores.json
        These scores are used later in evaluator.py to compare all models.

        Args:
            scores     : Dict of accuracy, precision, recall, f1
            model_name : Full name of the model
            model_key  : "lr", "rf", or "svm"

        Raises:
            RuntimeError: If saving fails
        """
        path = MODELS_DIR / f"{model_key}_scores.json"
        try:
            data = {
                "model_name": model_name,
                "model_key" : model_key,
                "accuracy"  : scores["accuracy"],
                "precision" : scores["precision"],
                "recall"    : scores["recall"],
                "f1"        : scores["f1"],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Scores saved to {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save scores: {e}")

    # -------------------------------------------------------------------------
    # SECTION 6: MAIN TRAIN FUNCTION
    # -------------------------------------------------------------------------

    def train(self, model_key: str):
        """
        Main function. Trains one model at a time.

        Flow:
            get model → fit → predict → compute scores
            → show scores → save model → save scores

        Args:
            model_key (str): "lr", "rf", or "svm"

        Returns:
            dict: {model_name, accuracy, precision, recall, f1}

        Raises:
            ValueError   : If model_key is invalid
            RuntimeError : If training or saving fails
        """
        logger.info("=" * 55)
        logger.info(f"Starting training: {model_key.upper()}")
        logger.info("=" * 55)

        # Step 1: Get model
        model_name, model = self._get_model(model_key)
        logger.info(f"Model    : {model_name}")
        logger.info(f"Train size: {self.X_train.shape[0]} samples")
        logger.info(f"Features  : {self.X_train.shape[1]}")

        # Step 2: Train model
        logger.info(f"Training {model_name}...")
        try:
            model.fit(self.X_train, self.y_train)
            logger.info(f"Training complete.")
        except Exception as e:
            raise RuntimeError(f"Training failed for {model_name}: {e}")

        # Step 3: Predict on test data
        try:
            y_pred = model.predict(self.X_test)
        except Exception as e:
            raise RuntimeError(f"Prediction failed for {model_name}: {e}")

        # Step 4: Compute scores
        scores = self._compute_scores(y_pred)

        # Step 5: Show scores
        self._show_scores(model_name, scores, y_pred)

        # Step 6: Save model
        self._save_model(model, model_key)

        # Step 7: Save scores
        self._save_scores(scores, model_name, model_key)

        logger.info(f"Training pipeline complete for {model_name}")

        return {
            "model_name": model_name,
            **scores
        }


# ── CLI / Quick Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.splitter import Splitter
    from src.feature_extractor import FeatureExtractor

    splitter  = Splitter()
    extractor = FeatureExtractor()

    try:
        # Step 1: Split
        X_train_text, X_test_text, y_train, y_test = splitter.split()

        # Step 2: Extract features
        X_train_tfidf, X_test_tfidf, y_train, y_test = extractor.extract(
            X_train_text, X_test_text, y_train, y_test
        )

        # Step 3: Train — one at a time
        trainer = Trainer(X_train_tfidf, X_test_tfidf, y_train, y_test)

        model_key = input("\nEnter model to train (lr / rf / svm): ").strip().lower()
        result    = trainer.train(model_key)

        print(f"\n✅ Training Complete!")
        print(f"{'=' * 50}")
        print(f"Model     : {result['model_name']}")
        print(f"Accuracy  : {result['accuracy']  * 100:.2f}%")
        print(f"Precision : {result['precision'] * 100:.2f}%")
        print(f"Recall    : {result['recall']    * 100:.2f}%")
        print(f"F1 Score  : {result['f1']        * 100:.2f}%")
        print(f"{'=' * 50}")

    except FileNotFoundError as e:
        print(f"\n File not found: {e}")

    except ValueError as e:
        print(f"\n Invalid input: {e}")

    except RuntimeError as e:
        print(f"\n Runtime error: {e}")

    except Exception as e:
        print(f"\n Unexpected error: {e}")