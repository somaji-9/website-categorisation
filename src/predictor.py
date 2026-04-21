# =============================================================================
# predictor.py
# Takes a single URL as input and returns its predicted category.
# Flow:
#   validate URL → fetch text → clean text → vectorize → predict → return result
# =============================================================================

import joblib
from pathlib import Path
from urllib.parse import urlparse

# Import fetcher and preprocessor
from src.fetcher import WebsiteFetcher
from src.preprocessor import TextPreprocessor

# Import paths from config
from config.config import MODEL_SAVE_PATH, VECTORIZER_SAVE_PATH

# Import logger
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class InvalidURLError(Exception):
    pass

class FetchError(Exception):
    pass

class InsufficientContentError(Exception):
    pass

class ModelNotFoundError(Exception):
    pass


# =============================================================================
# Predictor Class
# =============================================================================

class Predictor:

    def __init__(self):
        self.fetcher      = WebsiteFetcher()
        self.preprocessor = TextPreprocessor()
        self.vectorizer   = None
        self.model        = None

    # -------------------------------------------------------------------------
    # SECTION 1: LOAD MODELS
    # -------------------------------------------------------------------------

    def _load_vectorizer(self):
        """
        Loads vectorizer.pkl from disk.

        Raises:
            ModelNotFoundError: If vectorizer.pkl is missing
        """
        path = Path(VECTORIZER_SAVE_PATH)

        if not path.exists():
            raise ModelNotFoundError(
                "vectorizer.pkl not found. Please run feature_extractor.py first."
            )

        self.vectorizer = joblib.load(path)
        logger.info(f"Vectorizer loaded from: {path}")

    def _load_model(self):
        """
        Loads best_model.pkl from disk.

        Raises:
            ModelNotFoundError: If best_model.pkl is missing
        """
        path = Path(MODEL_SAVE_PATH)

        if not path.exists():
            raise ModelNotFoundError(
                "best_model.pkl not found. Please run evaluator.py first."
            )

        self.model = joblib.load(path)
        logger.info(f"Model loaded from: {path}")

    # -------------------------------------------------------------------------
    # SECTION 2: VALIDATE URL
    # -------------------------------------------------------------------------

    # def _validate_url(self, url):
    #     """
    #     Checks if the given URL has a valid format.

    #     Args:
    #         url (str): URL to validate

    #     Raises:
    #         InvalidURLError: If URL format is invalid
    #     """
    #     try:
    #         parsed = urlparse(url)
    #         if parsed.scheme not in ("http", "https") or not parsed.netloc:
    #             raise InvalidURLError(
    #                 f"'{url}' is not a valid URL. Please check and try again."
    #             )
    #     except InvalidURLError:
    #         raise
    #     except Exception:
    #         raise InvalidURLError(
    #             f"'{url}' is not a valid URL. Please check and try again."
    #         )

    # -------------------------------------------------------------------------
    # SECTION 3: PREDICT
    # -------------------------------------------------------------------------

    def predict(self, url):
        """
        Main function. Takes a URL and returns predicted category.

        Flow:
            validate URL
                ↓
            fetch + clean raw text (fetcher.py)
                ↓
            preprocess text (preprocessor.py)
                ↓
            load vectorizer.pkl → convert text to numbers
                ↓
            load best_model.pkl → predict category
                ↓
            return result dict

        Args:
            url (str): Website URL to classify

        Returns:
            dict: {
                "url"        : original URL,
                "domain"     : domain name,
                "category"   : predicted category,
                "confidence" : confidence score (0 to 1)
            }

        Raises:
            InvalidURLError          : If URL format is invalid
            FetchError               : If website could not be fetched
            InsufficientContentError : If not enough text found
            ModelNotFoundError       : If .pkl files are missing
        """
        logger.info("=" * 55)
        logger.info(f"Starting prediction for: {url}")
        logger.info("=" * 55)

        # Step 1: Validate URL
        #self._validate_url(url)
        #logger.info("URL validation passed.")

        # Step 2: Fetch website text
        try:
            domain, raw_text = self.fetcher.scrape_website(url)
            logger.info(f"Fetched text from: {domain}")

        except ValueError as e:
            raise InvalidURLError(
                f"'{url}' is not a valid URL. Please check and try again."
            )

        except RuntimeError as e:
            error_msg = str(e)

            if "NO_INTERNET" in error_msg:
                raise FetchError(
                    "No internet connection. Please check your network and try again."
                )
            elif "HOMEPAGE_FETCH_FAILED" in error_msg:
                raise FetchError(
                    f"Could not access {url}. "
                    "Website may be down or blocking requests. Try again later."
                )
            elif "INSUFFICIENT_CONTENT" in error_msg:
                raise InsufficientContentError(
                    f"Not enough content found on {url} (less than minimum words). "
                    "Cannot predict category."
                )
            elif "NO_CONTENT_EXTRACTED" in error_msg:
                raise InsufficientContentError(
                    f"No useful content found on {url}. "
                    "The page may require login or is empty."
                )
            else:
                raise FetchError(
                    f"Could not fetch {url}. Reason: {error_msg}"
                )

        # Step 3: Clean text using preprocessor
        try:
            cleaned_text = self.preprocessor.clean_text(raw_text)
            logger.info("Text preprocessing complete.")

        except ValueError:
            raise InsufficientContentError(
                f"Text became empty after cleaning for {url}. Cannot predict category."
            )

        # Step 4: Load vectorizer and model
        self._load_vectorizer()
        self._load_model()

        # Step 5: Convert text to TF-IDF numbers
        text_tfidf = self.vectorizer.transform([cleaned_text])
        logger.info("Text vectorized successfully.")

        # Step 6: Predict category
        try:
            prediction   = self.model.predict(text_tfidf)[0]
            probabilities = self.model.predict_proba(text_tfidf)[0]
            confidence   = round(float(max(probabilities)), 2)
            logger.info(f"Prediction: {prediction} | Confidence: {confidence}")

        except Exception as e:
            raise RuntimeError(
                f"Prediction failed for {url}. Please retrain the model. Error: {e}"
            )

        # Step 7: Return result
        result = {
            "url"        : url,
            "domain"     : domain,
            "category"   : prediction,
            "confidence" : confidence
        }

        logger.info(f"Prediction complete: {result}")
        return result


# =============================================================================
# Quick Test Mode
# Run directly to test predictor on a single URL:
#     python3 -m src.predictor
# =============================================================================

if __name__ == "__main__":
    predictor = Predictor()

    url = input("Enter URL to classify (e.g. https://www.cricbuzz.com): ").strip()

    try:
        result = predictor.predict(url)

        print(f"\n✅ Prediction Successful!")
        print(f"{'=' * 50}")
        print(f"URL        : {result['url']}")
        print(f"Domain     : {result['domain']}")
        print(f"Category   : {result['category']}")
        print(f"Confidence : {result['confidence'] * 100:.1f}%")
        print(f"{'=' * 50}")

    except InvalidURLError as e:
        print(f"\n Invalid URL: {e}")

    except FetchError as e:
        print(f"\n Fetch Failed: {e}")

    except InsufficientContentError as e:
        print(f"\n Insufficient Content: {e}")

    except ModelNotFoundError as e:
        print(f"\n Model Not Found: {e}")

    except Exception as e:
        print(f"\n Unexpected Error: {e}")