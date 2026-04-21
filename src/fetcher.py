# =============================================================================
# fetcher.py
# Fetches and cleans text from ONE website at a time.
# Handles URL normalization, page filtering, and Playwright fallback.
# Always starts scraping from root homepage regardless of URL entered.
# Used by dataset_builder.py to build the training dataset.
# =============================================================================

import re
from socket import create_connection 
import time
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# Import settings from config
from config.config import (
    FETCH_TIMEOUT,
    PLAYWRIGHT_TIMEOUT,
    PLAYWRIGHT_WAIT,
    USER_AGENT,
    MIN_TEXT_LENGTH,
    MIN_WORDS,
    MAX_WORDS,
    MAX_PAGES,
    CRAWL_DELAY,
    SKIP_URL_KEYWORDS
)

# Import logger
from src.logger import get_logger

class InvalidURLError(Exception):
    pass

logger = get_logger(__name__)

# =============================================================================
# WebsiteFetcher Class
# =============================================================================

class WebsiteFetcher:

    def __init__(self):
        # Browser-like headers to avoid getting blocked
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9"
        }

    # -------------------------------------------------------------------------
    # SECTION 1: URL NORMALIZER
    # Converts domain names and incomplete URLs to full valid URLs.
    # -------------------------------------------------------------------------

    def normalize_url(self, url):
        """
        Converts any URL format to a proper full URL.

        Examples:
            youtube.com        → https://youtube.com
            www.youtube.com    → https://www.youtube.com
            http://youtube.com → http://youtube.com  (kept as is)

        Args:
            url (str): Raw URL or domain name entered by user

        Returns:
            str: Normalized full URL
        """
        url = url.strip()

        # If URL doesn't start with http:// or https://, add https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
            logger.info(f"URL normalized to: {url}")

        return url

    # -------------------------------------------------------------------------
    # SECTION 2: ROOT URL EXTRACTOR
    # Always extracts root homepage URL from any given URL.
    # -------------------------------------------------------------------------

    def extract_root_url(self, url):
        """
        Extracts root homepage URL from any given URL.
        Always scrapes from root homepage for consistent dataset.

        Examples:
            https://youtube.com/about/team → https://youtube.com
            https://news.bbc.co.uk/sport   → https://news.bbc.co.uk
            https://youtube.com            → https://youtube.com (unchanged)

        Args:
            url (str): Any full URL

        Returns:
            str: Root homepage URL (scheme + domain only)
        """
        try:
            parsed = urlparse(url)

            # Rebuild URL with only scheme and domain
            # Removes any path, query params, fragments
            root_url = f"{parsed.scheme}://{parsed.netloc}"

            if root_url != url:
                logger.info(f"URL redirected to root homepage: {url} → {root_url}")

            return root_url

        except Exception as e:
            logger.error(f"Error extracting root URL from {url}: {e}")
            return url

    # -------------------------------------------------------------------------
    # SECTION 3: URL VALIDATION
    # -------------------------------------------------------------------------

    def extract_domain(self, url):
        """
        Validates URL and extracts domain name.

        Args:
            url (str): Full URL to validate

        Returns:
            str or None: Domain name if valid, None if invalid
        """
        try:
            parsed = urlparse(url)
            
            #TDL checking code 
            netloc = parsed.netloc
            has_valid_tld = "." in netloc and len(netloc.split(".")[-1]) >= 2

            # URL must have http/https scheme and a domain
            if parsed.scheme not in ("http", "https") or not netloc or not has_valid_tld:
                raise InvalidURLError(
                    f"'{url}' is not a valid URL. Please check and try again."
                )

            return parsed.netloc

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return None

    # -------------------------------------------------------------------------
    # SECTION 4: URL FILTER
    # Skips internal pages that are not useful for ML training.
    # -------------------------------------------------------------------------

    def is_valid_page(self, url):
        """
        Checks if an internal page is useful for ML training.
        Skips pages like privacy policy, terms, contact etc.
        Also skips anchor links like https://example.com#section
        To add more skip keywords → update SKIP_URL_KEYWORDS in config.py

        Args:
            url (str): Internal page URL to check

        Returns:
            bool: True if page is useful, False if should be skipped
        """
        try:
            parsed = urlparse(url)

            # Skip anchor only links like https://example.com#impact
            # These are not separate pages, just sections of same page
            if parsed.fragment and not parsed.path.strip("/"):
                logger.info(f"Skipping anchor link: {url}")
                return False

            # Skip unwanted pages using keywords from config
            url_lower = url.lower()
            for keyword in SKIP_URL_KEYWORDS:
                if keyword in url_lower:
                    logger.info(f"Skipping unwanted page: {url}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking URL {url}: {e}")
            return False

    # -------------------------------------------------------------------------
    # SECTION 5: WORD LIMIT HANDLER
    # Applies MIN and MAX word limits to extracted text.
    # -------------------------------------------------------------------------

    def apply_word_limit(self, text, url):
        """
        Applies word count limits to extracted text.

        Rules:
            Less than MIN_WORDS → raise error, skip website
            More than MAX_WORDS → cut to MAX_WORDS
            Between MIN and MAX → keep as is

        To change limits → update MIN_WORDS and MAX_WORDS in config.py

        Args:
            text (str): Extracted text from website
            url (str): Website URL (used for logging)

        Returns:
            str: Text within word limits

        Raises:
            RuntimeError: If text has less than MIN_WORDS words
        """
        words = text.split()
        word_count = len(words)

        logger.info(f"Word count before limit: {word_count} words")

        # Too less content → skip website
        if word_count < MIN_WORDS:
            raise RuntimeError(
                f"INSUFFICIENT_CONTENT: {url} has only {word_count} words "
                f"(minimum required: {MIN_WORDS})"
            )

        # Too much content → cut to MAX_WORDS
        if word_count > MAX_WORDS:
            text = " ".join(words[:MAX_WORDS])
            logger.info(f"Text cut from {word_count} to {MAX_WORDS} words")
            return text

        # Perfect range → keep as is
        logger.info(f"Word count is within limit: {word_count} words")
        return text

    # -------------------------------------------------------------------------
    # SECTION 6: FETCH PAGE
    # Tries requests first, falls back to Playwright if requests fails.
    # -------------------------------------------------------------------------

    def fetch_with_requests(self, url):
        """
        Fetches page HTML using requests library.
        Fast but may get blocked by some websites.

        Args:
            url (str): Page URL to fetch

        Returns:
            str or None: Raw HTML if successful, None if failed
        """
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=FETCH_TIMEOUT,
                allow_redirects=True,
                max_redirects=10
            )

            # Raises error if status is 4xx or 5xx
            response.raise_for_status()
            response.encoding = "utgf-8"
            logger.info(f"[requests] Successfully fetched: {url}")
            return response.content.decode("utf-8", errors="replace")

        except requests.exceptions.Timeout:
            logger.warning(f"[requests] Timeout for: {url}")
            return None

        except requests.exceptions.HTTPError as e:
            logger.warning(f"[requests] HTTP error {e.response.status_code} for: {url}")
            return None

        except requests.exceptions.ConnectionError:
            logger.warning(f"[requests] Connection error for: {url}")
            return None
        
        except requests.exceptions.TooManyRedirects:
            logger.warning(f"[requests] Too many redirects for: {url}")
            return None

        except Exception as e:
            logger.error(f"[requests] Unexpected error for {url}: {e}")
            return None

    def fetch_with_playwright(self, url, attempt=1):
        """
        Fetches page HTML using Playwright (real browser).
        Slower but works on websites that block requests.

        Args:
            url (str): Page URL to fetch
            attempt (int): Current attempt number (used for logging)

        Returns:
            str or None: Raw HTML if successful, None if failed
        """
        try:
            with sync_playwright() as p:
                # Launch headless chromium browser (no visible window)
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Go to URL, wait for page to load
                page.goto(url, timeout=PLAYWRIGHT_TIMEOUT)

                # Wait for JavaScript content to finish loading
                page.wait_for_timeout(PLAYWRIGHT_WAIT)

                html = page.content()
                browser.close()

            logger.info(f"[playwright] Attempt {attempt}/3 succeeded for: {url}")
            return html

        except TimeoutError:
            logger.warning(f"[playwright] Attempt {attempt}/3 → Timeout for: {url}")
            return None

        except Exception as e:
            # Catch specific playwright errors by message
            error_msg = str(e).lower()

            if "net::err_name_not_resolved" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → DNS error (domain not found): {url}")

            elif "net::err_connection_refused" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → Connection refused: {url}")

            elif "net::err_connection_timed_out" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → Connection timed out: {url}")

            elif "net::err_ssl" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → SSL certificate error: {url}")

            elif "browser" in error_msg or "chromium" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → Browser/Chromium crash: {url}")

            elif "page.goto" in error_msg or "navigation" in error_msg:
                logger.error(f"[playwright] Attempt {attempt}/3 → Navigation failed: {url}")

            else:
                logger.error(f"[playwright] Attempt {attempt}/3 → Unexpected error for {url}: {e}")

            return None

    def fetch_page(self, url):
        """
        Main fetch function with fallback.
        Tries requests first → falls back to Playwright with 3 retries if failed.

        Flow:
            requests → 
            if fail → Playwright attempt 1 → 
            if fail → Playwright attempt 2 → 
            if fail → Playwright attempt 3 → give up

        Args:
            url (str): Page URL to fetch

        Returns:
            str or None: Raw HTML if successful, None if all methods failed
        """
        try:
            create_connection(("8.8.8.8", 53), timeout=5)
        except OSError:
            raise RuntimeError("NO_INTERNET: No internet connection. Please check your network.")

        logger.info(f"Fetching: {url}")

        # Step 1: Try requests first (faster)
        html = self.fetch_with_requests(url)
        if html:
            return html

        # Step 2: Requests failed → try Playwright with 3 attempts
        logger.warning(f"Requests failed, switching to Playwright for: {url}")

        for attempt in range(1, 4):
            logger.info(f"[playwright] Starting attempt {attempt}/3 for: {url}")

            html = self.fetch_with_playwright(url, attempt=attempt)
            if html:
                return html

            # Wait before retrying (except after last attempt)
            if attempt < 3:
                wait_time = attempt * 2      # attempt 1 → 2s, attempt 2 → 4s
                logger.info(f"[playwright] Waiting {wait_time}s before attempt {attempt + 1}/3...")
                time.sleep(wait_time)

        # Step 3: All methods failed
        logger.error(f"All fetch attempts failed for: {url}")
        return None
    # -------------------------------------------------------------------------
    # SECTION 7: CLEAN HTML
    # Extracts clean meaningful text from raw HTML.
    # -------------------------------------------------------------------------

    def clean_html(self, html):
        """
        Extracts and cleans text from raw HTML.
        Removes scripts, styles, navigation and other noise.

        Args:
            html (str): Raw HTML content

        Returns:
            str: Cleaned plain text
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove tags that don't have useful content
            for tag in soup(["script", "style", "noscript",
                             "header", "footer", "nav",
                             "aside", "form", "button"]):
                tag.decompose()

            # Extract text from meaningful content tags only
            paragraphs = soup.find_all(["p", "li", "h1", "h2", "h3", "article"])
            text_blocks = []

            for p in paragraphs:
                text = p.get_text(strip=True)
                # Only keep blocks with meaningful content
                if text and len(text) > 40:
                    text_blocks.append(text)
                    
            if not text_blocks:
                logger.warning("No content tags found, falling back to body text.")
                body = soup.find("body")
                if body:
                    text_blocks = [body.get_text(strip=True)]
                    
            clean_text = " ".join(text_blocks)

            # Keep letters, numbers and spaces
            # Numbers are kept → useful for finance, sports categories
            
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)
            
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            return clean_text
        except MemoryError:
            logger.error("MemoryError: Page too large to process. Close other apps and retry.")
            return ""
        
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return ""
        

    # -------------------------------------------------------------------------
    # SECTION 8: EXTRACT INTERNAL LINKS
    # Gets useful links from same domain only.
    # -------------------------------------------------------------------------

    def extract_internal_links(self, html, base_url, domain):
        """
        Extracts internal links from a page.
        Filters out unwanted pages and anchor links.

        Args:
            html (str): Raw HTML content
            base_url (str): Base URL of the website
            domain (str): Domain to filter links

        Returns:
            list: List of valid internal URLs
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            links = set()

            for tag in soup.find_all("a", href=True):
                href = tag["href"]

                # Skip pure anchor links like #section, #top
                if href.startswith("#"):
                    continue

                # Convert relative URL to absolute URL
                # Example: /about → https://example.com/about
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)

                # Only keep links from same domain
                if parsed.netloc == domain:
                    # Filter out unwanted pages and anchor links
                    if self.is_valid_page(full_url):
                        links.add(full_url)

            logger.info(f"Found {len(links)} valid internal links on {base_url}")
            return list(links)

        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return []

    # -------------------------------------------------------------------------
    # SECTION 9: MAIN SCRAPE FUNCTION
    # Combines everything above to scrape one website.
    # -------------------------------------------------------------------------

    def scrape_website(self, url):
        """
        Main function to scrape a single website.
        Always starts from root homepage for consistent dataset.
        Called by dataset_builder.py for each URL.

        Flow:
            normalize_url()    → adds https:// if missing
            extract_root_url() → always goes to root homepage
            extract_domain()   → validates URL
            fetch homepage     → must succeed, else skip
            crawl pages        → up to MAX_PAGES internal pages
            apply_word_limit() → min 500, max 2000 words

        To scrape more pages → increase MAX_PAGES in config.py
        To skip more pages   → add keywords to SKIP_URL_KEYWORDS in config.py
        To change word limit → update MIN_WORDS and MAX_WORDS in config.py

        Args:
            url (str): Website URL or domain name

        Returns:
            tuple: (domain, clean_text)

        Raises:
            ValueError: If URL is invalid
            RuntimeError: If homepage fetch fails, no text extracted,
                         or word count is below minimum
        """

        # Step 1: Normalize URL (handles domain names like youtube.com)
        url = self.normalize_url(url)

        # Step 2: Extract root URL (always scrape from homepage)
        # Example: https://youtube.com/about → https://youtube.com
        url = self.extract_root_url(url)

        # Step 3: Validate URL
        domain = self.extract_domain(url)
        if not domain:
            raise ValueError(f"INVALID_URL: {url}")

        logger.info(f"Starting scrape for: {url}")

        # Step 4: Fetch homepage
        # Homepage MUST succeed → otherwise skip entire website
        homepage_html = self.fetch_page(url)
        if not homepage_html:
            raise RuntimeError(f"HOMEPAGE_FETCH_FAILED: {url}")

        # Step 5: Clean homepage text
        aggregated_text = self.clean_html(homepage_html)
        logger.info(f"Homepage text length: {len(aggregated_text)} characters")

        # Step 6: Extract valid internal links
        links = self.extract_internal_links(homepage_html, url, domain)

        # Limit to MAX_PAGES from config
        # To crawl more pages → increase MAX_PAGES in config.py
        links = links[:MAX_PAGES]
        logger.info(f"Will crawl {len(links)} internal pages")

        # Step 7: Fetch and clean each internal page
        for link in links:
            # Delay between requests to avoid getting blocked
            # To increase delay → change CRAWL_DELAY in config.py
            time.sleep(CRAWL_DELAY)

            page_html = self.fetch_page(link)

            if not page_html:
                # Internal page failed → skip it, continue with others
                logger.warning(f"Skipping internal page: {link}")
                continue

            page_text = self.clean_html(page_html)
            aggregated_text += " " + page_text
            logger.info(f"Added text from: {link}")

        # Step 8: Final cleanup
        aggregated_text = re.sub(r'\s+', ' ', aggregated_text).strip()

        # Step 9: Check minimum character length
        if len(aggregated_text) < MIN_TEXT_LENGTH:
            raise RuntimeError(f"NO_CONTENT_EXTRACTED: {url}")

        # Step 10: Apply word limits
        # Less than MIN_WORDS → raises RuntimeError → website skipped
        # More than MAX_WORDS → text cut to MAX_WORDS
        aggregated_text = self.apply_word_limit(aggregated_text, url)

        logger.info(f"Scraping complete for {url}")
        return domain, aggregated_text


# =============================================================================
# Quick Test Mode
# Run directly to test fetcher on a single URL:
#     python3 -m src.fetcher
# Results are only printed, not saved to CSV.
# To save data → use dataset_builder.py
# =============================================================================

if __name__ == "__main__":
    fetcher = WebsiteFetcher()

    # Accept URL or domain name from user
    url = input("Enter URL or domain (e.g. youtube.com): ").strip()

    try:
        domain, text = fetcher.scrape_website(url)
        word_count = len(text.split())

        print(f"\n✅ Scraping Successful!")
        print(f"{'='*50}")
        print(f"Domain      : {domain}")
        print(f"Word Count  : {word_count} words")
        print(f"Text Length : {len(text)} characters")
        print(f"{'='*50}")
        print(f"Sample Text :\n{text[:500]}")
        print(f"{'='*50}")

    except ValueError as e:
        print(f"\n Invalid URL: {e}")

    except RuntimeError as e:
        print(f"\n Scraping Failed: {e}")

    except Exception as e:
        print(f"\n Unexpected Error: {e}")
