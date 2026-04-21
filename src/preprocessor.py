# =============================================================================
# preprocessor.py
# Reads raw_data.csv, cleans text using NLP techniques,
# saves cleaned data to processed_data.csv.
# Cleaning steps:
#   1. Lowercase
#   2. Remove numbers
#   3. Remove stopwords (NLTK + custom list from config.py)
#   4. Lemmatization
# =============================================================================

import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import settings from config
from config.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    CUSTOM_STOPWORDS
)

# Import logger
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Download required NLTK data
# This runs only once — downloads language data needed for
# stopwords and lemmatization.
# If already downloaded, NLTK skips the download automatically.
# =============================================================================

def download_nltk_data():
    """
    Downloads required NLTK language data.
    Runs only once — skips if already downloaded.
    """
    required_packages = [
        "wordnet",      # needed for lemmatization
        "stopwords",    # needed for stopword removal
        "omw-1.4"       # needed for wordnet to work correctly
    ]

    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            logger.info(f"NLTK package ready: {package}")
        except Exception as e:
            logger.error(f"Failed to download NLTK package {package}: {e}")


# Download on import
download_nltk_data()


# =============================================================================
# TextPreprocessor Class
# =============================================================================

class TextPreprocessor:

    def __init__(self):
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Combine NLTK stopwords with custom stopwords from config
        # set() is used for faster lookup
        nltk_stopwords    = set(stopwords.words("english"))
        custom_stopwords  = set(CUSTOM_STOPWORDS)
        self.all_stopwords = nltk_stopwords.union(custom_stopwords)

        logger.info(f"Stopwords loaded: {len(nltk_stopwords)} NLTK + "
                    f"{len(custom_stopwords)} custom = "
                    f"{len(self.all_stopwords)} total")

    # -------------------------------------------------------------------------
    # SECTION 1: CLEANING STEPS
    # Each step is a separate method for clarity and easy future changes.
    # -------------------------------------------------------------------------

    def to_lowercase(self, text):
        """
        Converts all text to lowercase.
        Ensures words like 'Sports' and 'sports' are treated as same word.

        Args:
            text (str): Raw text

        Returns:
            str: Lowercased text
        """
        return text.lower()

    def remove_numbers(self, text):
        """
        Removes all numbers from text.
        Numbers like 123, 2024 don't help model learn category patterns.

        Args:
            text (str): Text with numbers

        Returns:
            str: Text without numbers
        """
        # Replace numbers with space
        text = re.sub(r'\d+', ' ', text)

        # Clean extra whitespace created after removal
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text):
        """
        Removes stopwords from text.
        Uses NLTK English stopwords + custom stopwords from config.py.
        To add more custom stopwords → update CUSTOM_STOPWORDS in config.py

        Args:
            text (str): Text with stopwords

        Returns:
            str: Text without stopwords
        """
        words        = text.split()
        filtered     = [word for word in words if word not in self.all_stopwords]
        return " ".join(filtered)

    def lemmatize(self, text):
        """
        Converts words to their correct root form.
        Example: playing → play, studies → study, better → good

        Args:
            text (str): Text to lemmatize

        Returns:
            str: Lemmatized text
        """
        words       = text.split()
        lemmatized  = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)

    # -------------------------------------------------------------------------
    # SECTION 2: MAIN CLEAN FUNCTION
    # Applies all cleaning steps in order.
    # -------------------------------------------------------------------------

    def clean_text(self, text):
        """
        Applies all cleaning steps in order:
            1. Lowercase
            2. Remove numbers
            3. Remove stopwords
            4. Lemmatization

        Args:
            text (str): Raw text from fetcher

        Returns:
            str: Fully cleaned text

        Raises:
            ValueError: If text is empty after cleaning
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text is empty or invalid")

        # Step 1: Lowercase
        text = self.to_lowercase(text)
        logger.debug(f"After lowercase: {len(text.split())} words")

        # Step 2: Remove numbers
        text = self.remove_numbers(text)
        logger.debug(f"After number removal: {len(text.split())} words")

        # Step 3: Remove stopwords
        text = self.remove_stopwords(text)
        logger.debug(f"After stopword removal: {len(text.split())} words")

        # Step 4: Lemmatization
        text = self.lemmatize(text)
        logger.debug(f"After lemmatization: {len(text.split())} words")

        # Final check — ensure cleaned text is not empty
        if not text.strip():
            raise ValueError("Text is empty after cleaning")

        return text

    # -------------------------------------------------------------------------
    # SECTION 3: PROCESS FULL DATASET
    # Reads raw_data.csv, cleans all text, saves to processed_data.csv.
    # -------------------------------------------------------------------------

    def process_dataset(self):
        """
        Reads raw_data.csv, cleans text for each row,
        saves cleaned data to processed_data.csv.

        Flow:
            Read raw_data.csv
                ↓
            For each row → clean text
                ↓
            Save to processed_data.csv
                ↓
            Show summary
        """
        logger.info("="*50)
        logger.info("Starting preprocessing")
        logger.info("="*50)

        # Step 1: Check raw data file exists
        if not os.path.exists(RAW_DATA_PATH):
            logger.error(f"raw_data.csv not found at: {RAW_DATA_PATH}")
            print(f"\n raw_data.csv not found. Please run dataset_builder first.")
            return

        # Step 2: Read raw data
        try:
            raw_df = pd.read_csv(RAW_DATA_PATH)
            logger.info(f"Loaded {len(raw_df)} rows from raw_data.csv")
        except Exception as e:
            logger.error(f"Error reading raw_data.csv: {e}")
            print(f"\n Error reading raw_data.csv: {e}")
            return

        # Step 3: Validate required columns
        required_columns = ["domain", "category", "text"]
        for col in required_columns:
            if col not in raw_df.columns:
                logger.error(f"Missing column in raw_data.csv: {col}")
                print(f"\n Missing column in raw_data.csv: {col}")
                return

        # Step 4: Process each row
        processed_rows = []
        success        = 0
        failed         = 0

        for index, row in raw_df.iterrows():
            domain   = row["domain"]
            category = row["category"]
            raw_text = row["text"]

            try:
                cleaned_text = self.clean_text(raw_text)
                processed_rows.append({
                    "domain"  : domain,
                    "category": category,
                    "text"    : cleaned_text
                })
                success += 1
                logger.info(f"Cleaned: {domain} | {category}")

            except ValueError as e:
                logger.warning(f"Skipping {domain}: {e}")
                failed += 1

            except Exception as e:
                logger.error(f"Unexpected error for {domain}: {e}")
                failed += 1

        # Step 5: Save processed data
        if processed_rows:
            try:
                processed_df = pd.DataFrame(processed_rows)

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

                processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
                logger.info(f"Saved {len(processed_rows)} rows to processed_data.csv")

            except Exception as e:
                logger.error(f"Error saving processed_data.csv: {e}")
                print(f"\n Error saving processed data: {e}")
                return
        else:
            logger.error("No rows to save after preprocessing")
            print("\n No data to save after preprocessing.")
            return

        # Step 6: Show summary
        self.show_summary(len(raw_df), success, failed)

    # -------------------------------------------------------------------------
    # SECTION 4: SUMMARY
    # -------------------------------------------------------------------------

    def show_summary(self, total, success, failed):
        """
        Displays final summary after preprocessing is complete.
        """
        print(f"\n{'='*40}")
        print(f"   PREPROCESSING SUMMARY")
        print(f"{'='*40}")
        print(f"  Total rows processed : {total}")
        print(f"  Successfully cleaned : {success}")
        print(f"  Failed               : {failed}")
        print(f"  Saved to             : {PROCESSED_DATA_PATH}")
        print(f"{'='*40}")

        logger.info(f"Preprocessing complete. Total: {total} | Success: {success} | Failed: {failed}")


# =============================================================================
# Run preprocessor
#     python3 -m src.preprocessor
# =============================================================================

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    preprocessor.process_dataset()