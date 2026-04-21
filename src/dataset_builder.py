# =============================================================================
# dataset_builder.py
# Reads URLs from urls_to_scrape.csv, calls fetcher for each URL,
# checks duplicates, saves scraped data to raw_data.csv.
# Run this file to build the training dataset.
# =============================================================================

import os
import csv
import pandas as pd

# Import fetcher
from src.fetcher import WebsiteFetcher

# Import settings from config
from config.config import (
    RAW_DATA_PATH,
    URLS_TO_SCRAPE_PATH,
)

# Import logger
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# DatasetBuilder Class
# =============================================================================

class DatasetBuilder:

    def __init__(self):
        self.fetcher = WebsiteFetcher()

        # Summary counters
        self.total     = 0
        self.saved     = 0
        self.skipped   = 0
        self.failed    = 0

    # -------------------------------------------------------------------------
    # SECTION 1: CSV HANDLERS
    # Read input URLs and write output data.
    # -------------------------------------------------------------------------

    def read_urls_csv(self):
        """
        Reads URLs and categories from urls_to_scrape.csv.

        Expected CSV format:
            url, category
            youtube.com, entertainment
            bbc.com, news

        Returns:
            list of tuples: [(url, category), ...]
            Empty list if file not found or empty
        """
        if not os.path.exists(URLS_TO_SCRAPE_PATH):
            logger.error(f"URL list file not found at: {URLS_TO_SCRAPE_PATH}")
            return []

        urls = []
        try:
            with open(URLS_TO_SCRAPE_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url      = row.get("url", "").strip()
                    category = row.get("category", "").strip()

                    # Skip empty rows
                    if not url or not category:
                        logger.warning(f"Skipping empty row in CSV: {row}")
                        continue

                    urls.append((url, category))

            logger.info(f"Read {len(urls)} URLs from {URLS_TO_SCRAPE_PATH}")
            return urls

        except Exception as e:
            logger.error(f"Error reading URLs CSV: {e}")
            return []

    def load_existing_domains(self):
        """
        Loads already scraped domains from raw_data.csv.
        Used for duplicate checking.

        Returns:
            set: Set of domain names already in raw_data.csv
        """
        if not os.path.exists(RAW_DATA_PATH):
            logger.info("raw_data.csv not found. Starting fresh dataset.")
            return set()

        try:
            df = pd.read_csv(RAW_DATA_PATH)

            # Check if domain column exists
            if "domain" not in df.columns:
                logger.warning("No domain column found in raw_data.csv")
                return set()

            domains = set(df["domain"].dropna().unique())
            logger.info(f"Found {len(domains)} existing domains in raw_data.csv")
            return domains

        except Exception as e:
            logger.error(f"Error reading raw_data.csv: {e}")
            return set()

    def save_to_csv(self, domain, category, text):
        """
        Saves scraped data to raw_data.csv.
        Creates file with headers if it doesn't exist or is empty.
        Appends to existing file if it has data.

        Args:
            domain (str): Website domain name
            category (str): Website category
            text (str): Cleaned website text
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

            # Check if file is empty or doesn't exist
            # This fixes the bug where setup_project.py creates empty file
            file_is_empty = (
                not os.path.exists(RAW_DATA_PATH) or
                os.path.getsize(RAW_DATA_PATH) == 0
            )

            with open(RAW_DATA_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write headers only if file is empty or new
                if file_is_empty:
                    writer.writerow(["domain", "category", "text"])
                    logger.info("Created new raw_data.csv with headers")

                writer.writerow([domain, category, text])

            logger.info(f"Saved to CSV: {domain} | {category}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

    # -------------------------------------------------------------------------
    # SECTION 2: MAIN BUILD FUNCTION
    # Orchestrates the entire dataset building process.
    # -------------------------------------------------------------------------

    def build(self):
        """
        Main function to build the dataset.
        Reads URLs, fetches text, checks duplicates, saves to CSV.

        Flow:
            Read urls_to_scrape.csv
                ↓
            For each URL:
                normalize → extract domain → check duplicate
                → fetch → save to raw_data.csv
                ↓
            Show summary
        """
        logger.info("="*50)
        logger.info("Starting dataset build process")
        logger.info("="*50)

        # Step 1: Read URLs from input CSV
        url_list = self.read_urls_csv()
        if not url_list:
            logger.error("No URLs to process. Please check urls_to_scrape.csv")
            print("\n  No URLs found. Please check your urls_to_scrape.csv file.")
            return

        self.total = len(url_list)
        logger.info(f"Total URLs to process: {self.total}")

        # Step 2: Load already scraped domains for duplicate check
        existing_domains = self.load_existing_domains()

        # Step 3: Process each URL
        for url, category in url_list:
            logger.info(f"Processing: {url} | Category: {category}")

            try:
                # Normalize and extract domain for duplicate check
                # Reusing fetcher methods for consistency
                normalized = self.fetcher.normalize_url(url)
                root_url   = self.fetcher.extract_root_url(normalized)
                domain     = self.fetcher.extract_domain(root_url)

                if not domain:
                    logger.warning(f"Could not extract domain from: {url}")
                    self.failed += 1
                    continue

                # Step 4: Check for duplicate
                if domain in existing_domains:
                    logger.warning(f"Duplicate found, skipping: {domain}")
                    self.skipped += 1
                    continue

                # Step 5: Fetch website text
                fetched_domain, text = self.fetcher.scrape_website(url)

                # Step 6: Save to CSV
                self.save_to_csv(fetched_domain, category, text)

                # Add to existing domains to prevent duplicates
                # within same run as well
                existing_domains.add(fetched_domain)
                self.saved += 1

            except ValueError as e:
                logger.error(f"Invalid URL {url}: {e}")
                self.failed += 1

            except RuntimeError as e:
                logger.error(f"Scraping failed for {url}: {e}")
                self.failed += 1

            except Exception as e:
                logger.error(f"Unexpected error for {url}: {e}")
                self.failed += 1

        # Step 7: Show summary
        self.show_summary()

    # -------------------------------------------------------------------------
    # SECTION 3: SUMMARY
    # -------------------------------------------------------------------------

    def show_summary(self):
        """
        Displays final summary after dataset build is complete.
        """
        print(f"\n{'='*40}")
        print(f"   DATASET BUILD SUMMARY")
        print(f"{'='*40}")
        print(f"  Total URLs given    : {self.total}")
        print(f"  Successfully saved  : {self.saved}")
        print(f"  Skipped duplicates  : {self.skipped}")
        print(f"  Failed              : {self.failed}")
        print(f"{'='*40}")

        logger.info("Dataset build complete.")
        logger.info(f"Total: {self.total} | Saved: {self.saved} | Skipped: {self.skipped} | Failed: {self.failed}")


# =============================================================================
# Run dataset builder
#     python3 -m src.dataset_builder
# =============================================================================

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()