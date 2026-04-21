# config.py
# Central configuration file for website-classifier project.
# All settings are defined here. Change values here only, not in other files.
# =============================================================================

import os

# =============================================================================
# 1. PATH SETTINGS
# All folder and file paths used in the project.
# If you move your project to a different location, only change BASE_DIR here.
# =============================================================================

# Root directory of the project (automatically detected)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw", "raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

# Model paths
MODEL_SAVE_PATH      = os.path.join(BASE_DIR, "models", "saved", "best_model.pkl")
VECTORIZER_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved", "vectorizer.pkl")

# Log path
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "app.log")


# =============================================================================
# 2. CATEGORIES
# List of all website categories the model will classify.
# To add a new category: just add it to this list.
# To remove a category: remove it from this list.
# NOTE: After changing categories, you must re-collect data and re-train model.
# =============================================================================

CATEGORIES = [
    "ecommerce",
    "education",
    "news",
    "technology",
    "health",
    "finance",
    "tourism",
    "entertainment",
    "sports",
    "government",
    "social_media",
    "legal"
]

# Total number of categories (automatically counted, no need to change)
NUM_CATEGORIES = len(CATEGORIES)


# =============================================================================
# 3. FETCHER SETTINGS
# Settings for fetching website text.
# =============================================================================

# Maximum time (in seconds) to wait for a website to respond
# Increase this if you get too many timeout errors on slow websites
FETCH_TIMEOUT = 10

# Maximum time (in milliseconds) Playwright waits for a page to load
# Increase if JavaScript-heavy websites are not loading fully
PLAYWRIGHT_TIMEOUT = 15000

# Time (in milliseconds) Playwright waits after page loads for JS to finish
# Increase if website content is missing in fetched text
PLAYWRIGHT_WAIT = 2500

# Browser header sent with requests to look like a real browser
# This helps avoid getting blocked by websites
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Minimum number of characters a fetched text must have to be considered valid
# Pages with less text than this will be skipped
MIN_TEXT_LENGTH = 100


# =============================================================================
# 4. DATA SETTINGS
# Settings for preparing and splitting data for training.
# =============================================================================

# Fraction of data used for testing (0.2 means 20% test, 80% training)
# Increase (e.g. 0.3) if you want more testing data
# Decrease (e.g. 0.1) if you want more training data
TEST_SIZE = 0.2

# Random state ensures same split every time you run the code
# Change this number to get a different random split
# Keep it fixed during experiments so results are comparable
RANDOM_STATE = 42

# Maximum number of text features (words) to extract from websites
# Higher = more detail but slower training
# Lower = faster but may lose important words
MAX_FEATURES = 10000


# =============================================================================
# 5. MODEL SETTINGS
# Settings for each ML model.
# All 3 models will be trained and compared.
# Best model will be saved automatically.
# =============================================================================

MODEL_SETTINGS = {

    "logistic_regression": {
        # Maximum number of iterations the model tries to find best solution
        # Increase if you see "did not converge" warning during training
        "max_iter": 1000,

        # Strength of regularization (prevents overfitting)
        # Smaller value = stronger regularization
        # Try: 0.01, 0.1, 1, 10 → higher usually means better fit on training data
        "C": 1.0,

        # Keeps results same every run
        "random_state": RANDOM_STATE,
    },

    "random_forest": {
        # Number of decision trees to build
        # More trees = better accuracy but slower training
        # Try: 50, 100, 200
        "n_estimators": 100,

        # Maximum depth of each tree
        # None = trees grow until all leaves are pure (may overfit)
        # Try: 10, 20, 50, None
        "max_depth": None,

        # Minimum samples required to split a node
        # Higher = simpler trees (less overfitting)
        # Try: 2, 5, 10
        "min_samples_split": 2,

        # Keeps results same every run
        "random_state": RANDOM_STATE,
    },

    "svm": {
        # Type of decision boundary
        # "linear"  → good for text classification (recommended)
        # "rbf"     → good for complex patterns but slower
        # "poly"    → polynomial boundary
        "kernel": "linear",

        # Strength of regularization (similar to logistic regression C)
        # Higher value = model tries harder to classify training data correctly
        # Try: 0.1, 1, 10
        "C": 1.0,

        # Keeps results same every run
        "random_state": RANDOM_STATE,
    }
}


# =============================================================================
# 6. LOGGING SETTINGS
# Settings for how logs are recorded.
# =============================================================================

# Log level controls what gets recorded in the log file
# "DEBUG"   → records everything (use during development)
# "INFO"    → records important steps only (use during normal use)
# "WARNING" → records only warnings and errors
# "ERROR"   → records only errors
LOG_LEVEL = "DEBUG"

# Format of each log line saved in app.log
# %(asctime)s   → date and time
# %(levelname)s → DEBUG / INFO / WARNING / ERROR
# %(filename)s  → which file created this log
# %(message)s   → actual log message
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(filename)s | %(message)s"

# Date format used in log lines
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# Maximum number of internal pages to scrape per website
# Increase this value if you want more text per website
MAX_PAGES = 5

# Delay in seconds between page requests
# Increase if websites are blocking you
CRAWL_DELAY = 1
# Pages to skip during crawling
# Add more keywords here if needed in future
SKIP_URL_KEYWORDS = [
    "privacy", "terms", "copyright",
    "about", "contact", "legal",
    "cookies", "disclaimer", "sitemap"
]

# Minimum words required per website
# Less than this → website is skipped
MIN_WORDS = 400

# Maximum words to save per website
# More than this → text is cut to this limit
MAX_WORDS = 2000

# Path to input URL list file
URLS_TO_SCRAPE_PATH = os.path.join(BASE_DIR, "data", "urls", "urls_to_scrape.csv")

# Custom stopwords to remove during preprocessing
# Add more words here if needed in future
CUSTOM_STOPWORDS = [
    "ok", "hi", "eg", "ie", "vs",
    "mr", "dr", "mrs", "ms",
    "etc", "via", "per", "non"
]




# Total unique values: 16

# Value counts:
#  Category
# Education                          114
# Business/Corporate                 109
# Travel                             107
# Streaming Services                 105
# Sports                             104
# E-Commerce                         102
# Games                               98
# News                                96
# Health and Fitness                  96
# Photography                         93
# Computers and Technology            93
# Food                                92
# Law and Government                  84
# Social Networking and Messaging     83
# Forums                              16
# Adult                               16
# Name: count, dtype: int64

# CATEGORIES = [
    # "ecommerce",
    # "education",
    # "news",
    # "technology",
    # "health",
    # "finance",
    # "tourism",
    # "entertainment",
    # "sports",
    # "government",
    # "social_media",
    # "legal"
# ]
