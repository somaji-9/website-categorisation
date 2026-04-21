"""
splitter.py

Responsibilities:
- Load processed_data.csv
- Validate data
- Split into 80% train / 20% test
- Returns raw text splits (NO TF-IDF here)

Flow:
    processed_data.csv
        → load + validate
        → split (stratified)
        → return X_train_text, X_test_text, y_train, y_test

Note:
    TF-IDF is NOT done here.
    Raw text is returned so feature_extractor.py can fit
    TF-IDF on train only (prevents data leakage).
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config.config import TEST_SIZE, RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)

# ── Path ──────────────────────────────────────────────────────────────────────
PROCESSED_DATA_PATH = Path("data/processed/processed_data.csv")


# ── Splitter Class ────────────────────────────────────────────────────────────

class Splitter:

    # -------------------------------------------------------------------------
    # SECTION 1: LOAD DATA
    # -------------------------------------------------------------------------

    def load_data(self):
        """
        Loads processed_data.csv and validates it.

        Returns:
            pd.DataFrame: Validated dataframe

        Raises:
            FileNotFoundError : If processed_data.csv not found
            ValueError        : If required columns missing or data empty
        """
        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(
                f"processed_data.csv not found at: {PROCESSED_DATA_PATH}\n"
                f"Please run preprocessor.py first."
            )

        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            logger.info(f"Loaded {len(df)} rows from {PROCESSED_DATA_PATH}")
        except Exception as e:
            raise ValueError(f"Error reading processed_data.csv: {e}")

        # Validate required columns
        required_columns = {"domain", "category", "text"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in processed_data.csv: {missing}")

        # Validate not empty
        if df.empty:
            raise ValueError("processed_data.csv is empty. Run preprocessor.py first.")

        # Drop rows with missing text or category
        before = len(df)
        df = df.dropna(subset=["text", "category"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing text/category.")

        logger.info(f"Final dataset: {len(df)} rows | {df['category'].nunique()} categories")
        logger.info(f"Category distribution:\n{df['category'].value_counts().to_string()}")

        return df

    # -------------------------------------------------------------------------
    # SECTION 2: SPLIT DATA
    # -------------------------------------------------------------------------

    def split(self):
        """
        Main function.
        Loads processed_data.csv and splits into train/test sets.

        Split ratio : 80% train / 20% test (controlled by TEST_SIZE in config)
        Stratified  : ensures equal category distribution in both sets

        Returns:
            tuple: (X_train_text, X_test_text, y_train, y_test)
                X_train_text → Series of cleaned text for training
                X_test_text  → Series of cleaned text for testing
                y_train      → Series of category labels for training
                y_test       → Series of category labels for testing

        Raises:
            FileNotFoundError : If processed_data.csv not found
            ValueError        : If data is invalid or split fails
        """
        logger.info("=" * 55)
        logger.info("Starting data splitting...")
        logger.info("=" * 55)

        # Step 1: Load data
        df = self.load_data()

        X = df["text"]
        y = df["category"]

        # Step 2: Split
        # stratify=y → ensures each category has equal representation
        # in both train and test sets
        # Example: if legal=58 total → train=46, test=12 (same ratio)
        try:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                X, y,
                test_size    = TEST_SIZE,
                random_state = RANDOM_STATE,
                stratify     = y
            )
        except Exception as e:
            raise ValueError(f"Data splitting failed: {e}")

        # ── Summary ───────────────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("Splitting complete. Summary:")
        logger.info(f"  Total samples : {len(df)}")
        logger.info(f"  Train samples : {len(X_train_text)} ({int((1 - TEST_SIZE) * 100)}%)")
        logger.info(f"  Test samples  : {len(X_test_text)}  ({int(TEST_SIZE * 100)}%)")
        logger.info("=" * 55)

        return X_train_text, X_test_text, y_train, y_test


# ── CLI / Quick Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    splitter = Splitter()

    try:
        X_train, X_test, y_train, y_test = splitter.split()

        print(f"\n✅ Splitting Successful!")
        print(f"{'=' * 50}")
        print(f"Total samples  : {len(X_train) + len(X_test)}")
        print(f"Train samples  : {len(X_train)}")
        print(f"Test samples   : {len(X_test)}")
        print(f"{'=' * 50}")
        print(f"Train category distribution:\n{y_train.value_counts().to_string()}")
        print(f"{'=' * 50}")
        print(f"Test category distribution:\n{y_test.value_counts().to_string()}")
        print(f"{'=' * 50}")

    except FileNotFoundError as e:
        print(f"\n File not found: {e}")

    except ValueError as e:
        print(f"\n Data error: {e}")

    except Exception as e:
        print(f"\n Unexpected error: {e}")
