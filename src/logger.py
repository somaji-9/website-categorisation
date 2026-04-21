# =============================================================================
# logger.py
# Sets up logging for the entire project.
# Every other file imports get_logger() from here to record logs.
# All logs are saved to logs/app.log AND shown in terminal.
# =============================================================================

import logging
import os

# Import logging settings from config
from config.config import LOG_FILE_PATH, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the given name.

    How to use in any file:
        from src.logger import get_logger
        logger = get_logger(__name__)       
        logger.info("Your message here")

    Args:
        name (str): Name of the logger, usually pass __name__
                    This automatically uses the current filename as logger name.
                    Helps identify which file created which log.

    Returns:
        logging.Logger: Ready to use logger object
    """

    # Get or create a logger with the given name
    # If logger with this name already exists, it returns the same one
    # This prevents duplicate log entries
    logger = logging.getLogger(name)

    # If this logger already has handlers, it means it was already set up
    # So we skip setup to avoid duplicate logs
    if logger.handlers:
        return logger

    # Convert log level string from config to logging module constant
    # Example: "DEBUG" string → logging.DEBUG number
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.DEBUG)

    # Set the minimum level of messages this logger will handle
    logger.setLevel(log_level)

    # -------------------------------------------------------------------------
    # Create formatter
    # Formatter decides how each log line looks
    # Example output: 2024-01-15 10:30:45 | INFO | fetcher.py | Fetching URL...
    # -------------------------------------------------------------------------
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )

    # -------------------------------------------------------------------------
    # Handler 1: File Handler
    # Saves all logs to logs/app.log file
    # mode="a" means append → new logs are added, old logs are not deleted
    # -------------------------------------------------------------------------
    try:
        # Make sure the logs folder exists before creating the file
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

        file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except Exception as e:
        # If log file cannot be created, print warning but don't crash the project
        print(f"[logger] WARNING: Could not create log file at {LOG_FILE_PATH}. Reason: {e}")

    # -------------------------------------------------------------------------
    # Handler 2: Console Handler
    # Shows logs in terminal so you can see what's happening in real time
    # -------------------------------------------------------------------------
    
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.ERROR)
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return logger


# =============================================================================
# Quick test
# Run this file directly to check if logger is working:
#     python3 src/logger.py
# =============================================================================

if __name__ == "__main__":
    # Create a test logger
    logger = get_logger(__name__)

    # Test all log levels
    logger.debug("DEBUG: This is a debug message - shows detailed info")
    logger.info("INFO: This is an info message - shows normal progress")
    logger.warning("WARNING: This is a warning - something needs attention")
    logger.error("ERROR: This is an error - something went wrong")

    print(f"\n✅ Logger is working!")
    print(f"📄 Check your log file at: {LOG_FILE_PATH}")    