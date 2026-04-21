# main.py
# Entry point for Website Classifier.
# Takes URL from user and returns predicted category.
# =============================================================================
from halo import Halo   
from src.predictor import (
    Predictor,
    InvalidURLError,
    FetchError,
    InsufficientContentError,
    ModelNotFoundError
)

from src.logger import get_logger

logger = get_logger(__name__)


def main():

    predictor = Predictor()

    print("\n" + "=" * 50)
    print("        Website Category Classifier")
    print("=" * 50)

    while True:

        # Take URL from user
        url = input("\nEnter URL (or 'exit' to quit): ").strip()

        # Exit condition
        if url.lower() == "exit":
            print("\n exiting .....")
            break

        # Skip empty input
        if not url:
            print(" Please enter a URL.")
            continue

        # Predict
        try:
            
            spinner = Halo(spinner='dots')
            spinner.start()

            result = predictor.predict(url)

            spinner.stop()

            # Check confidence threshold
            if result['confidence'] < 0.50:
                print(f"\n  Not Confident Enough")
                print(f"{'=' * 50}")
                print(f"URL        : {result['url']}")
                print(f"Domain     : {result['domain']}")
                print(f"Confidence : {result['confidence'] * 100:.1f}%")
                print(f"Cannot reliably classify this website.")
                print(f"{'=' * 50}")
            else:
                print(f"\n Prediction Successful!")
                print(f"{'=' * 50}")
                print(f"URL        : {result['url']}")
                print(f"Domain     : {result['domain']}")
                print(f"Category   : {result['category']}")
                print(f"Confidence : {result['confidence'] * 100:.1f}%")
                print(f"{'=' * 50}")
                
                logger.info(f"Prediction: {result}")

        except InvalidURLError as e:
            print(f"\n Invalid URL: {e}")
            spinner.stop()
        except FetchError as e:
            print(f"\ Fetch Failed: {e}")
            spinner.stop()
        except InsufficientContentError as e:
            print(f"\n {e}")
            spinner.stop()
        except ModelNotFoundError as e:
            print(f"\  Model Not Found: {e}")
            spinner.stop()

        except Exception as e:
            print(f"\  Unexpected Error: {e}")
            logger.error(f"Unexpected error: {e}")
            spinner.stop()


# =============================================================================
# Run
#     python3 main.py
# =============================================================================

if __name__ == "__main__":
    main()