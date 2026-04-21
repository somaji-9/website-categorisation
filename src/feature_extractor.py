"""
feature_extractor.py

Responsibilities:
- Receive train/test text splits from splitter.py
- Fit TF-IDF on X_train_text only (prevents data leakage)
- Transform X_test_text using same fitted vectorizer
- Save vectorizer.pkl for later use in predictor.py
- Return X_train_tfidf, X_test_tfidf, y_train, y_test

Flow:
    X_train_text, X_test_text, y_train, y_test  ← from splitter.py
        → fit TF-IDF on X_train_text only
        → transform X_test_text
        → save vectorizer.pkl
        → return X_train_tfidf, X_test_tfidf, y_train, y_test

Note:
    TF-IDF is ONLY fitted on training data.
    Test data is only transformed, never fitted.
    This prevents data leakage.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from config.config import MAX_FEATURES
from src.logger import get_logger

logger = get_logger(__name__)

# ── Path ──────────────────────────────────────────────────────────────────────
VECTORIZER_PATH = Path("models/saved/vectorizer.pkl")


# ── FeatureExtractor Class ────────────────────────────────────────────────────

class FeatureExtractor:

    def __init__(self):
        # Initialize TF-IDF Vectorizer with settings from config
        # max_features → keep only top N words by TF-IDF score
        # ngram_range  → (1,2) extracts single words AND two-word phrases
        #                Example: "machine learning" treated as one feature
        # sublinear_tf → reduces effect of very frequent words
        #                Example: word appearing 100 times not 100x more important
        self.vectorizer = TfidfVectorizer(
            max_features = MAX_FEATURES,
            ngram_range  = (1, 2),
            sublinear_tf = True,
        )

    # -------------------------------------------------------------------------
    # SECTION 1: VALIDATE INPUT
    # -------------------------------------------------------------------------

    def _validate_input(self, X_train_text, X_test_text, y_train, y_test):
        """
        Validates input received from splitter.py.

        Args:
            X_train_text : Training text (pd.Series or list)
            X_test_text  : Testing text (pd.Series or list)
            y_train      : Training labels
            y_test       : Testing labels

        Raises:
            ValueError: If any input is empty or mismatched
        """
        if X_train_text is None or len(X_train_text) == 0:
            raise ValueError("X_train_text is empty. Check splitter.py output.")

        if X_test_text is None or len(X_test_text) == 0:
            raise ValueError("X_test_text is empty. Check splitter.py output.")

        if len(X_train_text) != len(y_train):
            raise ValueError(
                f"X_train_text ({len(X_train_text)}) and "
                f"y_train ({len(y_train)}) size mismatch."
            )

        if len(X_test_text) != len(y_test):
            raise ValueError(
                f"X_test_text ({len(X_test_text)}) and "
                f"y_test ({len(y_test)}) size mismatch."
            )

        logger.info(f"Input validation passed.")
        logger.info(f"  Train samples : {len(X_train_text)}")
        logger.info(f"  Test samples  : {len(X_test_text)}")

    # -------------------------------------------------------------------------
    # SECTION 2: SAVE VECTORIZER
    # -------------------------------------------------------------------------

    def _save_vectorizer(self):
        """
        Saves fitted TF-IDF vectorizer to models/saved/vectorizer.pkl.
        This file is required later in predictor.py to convert
        new URL text into the same number format as training data.

        Raises:
            RuntimeError: If saving fails
        """
        try:
            VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)
            logger.info(f"Vectorizer saved to: {VECTORIZER_PATH}")

        except Exception as e:
            raise RuntimeError(f"Failed to save vectorizer: {e}")

    # -------------------------------------------------------------------------
    # SECTION 3: EXTRACT FEATURES
    # -------------------------------------------------------------------------

    def extract(self, X_train_text, X_test_text, y_train, y_test):
        """
        Main function.
        Fits TF-IDF on train text → transforms both train and test
        → saves vectorizer.pkl → returns TF-IDF matrices.

        Args:
            X_train_text : Training text from splitter.py (pd.Series or list)
            X_test_text  : Testing text from splitter.py (pd.Series or list)
            y_train      : Training labels from splitter.py
            y_test       : Testing labels from splitter.py

        Returns:
            tuple: (X_train_tfidf, X_test_tfidf, y_train, y_test)
                X_train_tfidf → sparse TF-IDF matrix for training
                X_test_tfidf  → sparse TF-IDF matrix for testing
                y_train       → training labels (unchanged)
                y_test        → testing labels (unchanged)

        Raises:
            ValueError   : If input is invalid or TF-IDF fails
            RuntimeError : If saving vectorizer fails
        """
        logger.info("=" * 55)
        logger.info("Starting feature extraction...")
        logger.info("=" * 55)

        # Step 1: Validate input
        self._validate_input(X_train_text, X_test_text, y_train, y_test)

        # Step 2: Fit TF-IDF on training data ONLY
        # fit_transform() → learns vocabulary from X_train + converts to numbers
        # NEVER fit on X_test → would cause data leakage
        logger.info(f"Fitting TF-IDF on training data (max_features={MAX_FEATURES})...")
        try:
            X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
            logger.info(f"TF-IDF fit complete.")
            logger.info(f"  Vocabulary size  : {len(self.vectorizer.vocabulary_)}")
            logger.info(f"  Train matrix     : {X_train_tfidf.shape}")
            # Shape example: (744, 5000)
            # 744 websites, each represented as 5000 numbers

        except Exception as e:
            raise ValueError(f"TF-IDF fitting failed: {e}")

        # Step 3: Transform test data using SAME fitted vectorizer
        # transform() only → converts X_test using vocabulary learned from X_train
        # Does NOT learn anything new from X_test
        logger.info(f"Transforming test data using fitted vectorizer...")
        try:
            X_test_tfidf = self.vectorizer.transform(X_test_text)
            logger.info(f"  Test matrix      : {X_test_tfidf.shape}")

        except Exception as e:
            raise ValueError(f"TF-IDF transformation failed: {e}")

        # Step 4: Save vectorizer
        self._save_vectorizer()

        # ── Summary ───────────────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("Feature extraction complete. Summary:")
        logger.info(f"  Train matrix shape : {X_train_tfidf.shape}")
        logger.info(f"  Test matrix shape  : {X_test_tfidf.shape}")
        logger.info(f"  Vocabulary size    : {len(self.vectorizer.vocabulary_)}")
        logger.info(f"  Vectorizer saved   : {VECTORIZER_PATH}")
        logger.info("=" * 55)

        return X_train_tfidf, X_test_tfidf, y_train, y_test


# ── CLI / Quick Test ──────────────────────────────────────────────────────────
# Run directly to test feature extraction:
#     python3 -m src.feature_extractor

if __name__ == "__main__":
    from src.splitter import Splitter

    splitter  = Splitter()
    extractor = FeatureExtractor()

    try:
        # Step 1: Get splits from splitter
        X_train_text, X_test_text, y_train, y_test = splitter.split()

        # Step 2: Extract features
        X_train_tfidf, X_test_tfidf, y_train, y_test = extractor.extract(
            X_train_text, X_test_text, y_train, y_test
        )

        print(f"\n✅ Feature Extraction Successful!")
        print(f"{'=' * 50}")
        print(f"Train matrix shape : {X_train_tfidf.shape}")
        print(f"Test matrix shape  : {X_test_tfidf.shape}")
        print(f"Vocabulary size    : {len(extractor.vectorizer.vocabulary_)}")
        print(f"Vectorizer saved   : {VECTORIZER_PATH}")
        print(f"{'=' * 50}")

    except FileNotFoundError as e:
        print(f"\n File not found: {e}")

    except ValueError as e:
        print(f"\n Data error: {e}")

    except RuntimeError as e:
        print(f"\n Runtime error: {e}")

    except Exception as e:
        print(f"\n  Unexpected error: {e}")