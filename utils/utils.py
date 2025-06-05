import os
from datetime import datetime, timedelta
import logging
import colorlog
import dateparser
from typing import List, Dict, Any
import random
import torch
import argparse


def setup_logger(logging_level):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger()
    logger.setLevel(logging_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()

    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    return logger


def load_prompts_from_dir_into_dict(base_path):
    result = {}

    for root, _, files in os.walk(base_path):
        rel_path = os.path.relpath(root, base_path)

        if rel_path == ".":
            continue

        prefix = rel_path.replace("/", "_").replace("\\", "_")

        for file in files:
            if file.endswith(".txt"):
                key = f"{prefix}_{os.path.splitext(file)[0]}"

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    result[key] = content
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return result


def remove_newlines_from_text(final_response):
    return final_response.lstrip("\n")


def normalize_date(date_string):
    """
    Convert natural language date references to YYYY-MM-DD format.
    Uses dateparser for robust parsing of various date formats and expressions.

    Args:
        date_string (str): Natural language date string like "tomorrow", "next Friday", "04/15/25"

    Returns:
        str: Normalized date in YYYY-MM-DD format, or None if parsing fails
    """
    if not date_string:
        return None

    try:
        date_string = date_string.lower().strip()
        logging.info(f"Normalizing date: '{date_string}'")
        # Special case for "next" + weekday which dateparser sometimes gets wrong
        if date_string.startswith("next "):
            day_name = date_string.split("next ")[1].strip()
            if day_name in [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]:
                today = datetime.now()
                logging.info(f"Today is: {today.strftime('%Y-%m-%d')}")
                day_mapping = {
                    "monday": 0,
                    "tuesday": 1,
                    "wednesday": 2,
                    "thursday": 3,
                    "friday": 4,
                    "saturday": 5,
                    "sunday": 6,
                }
                target_day = day_mapping.get(day_name)

                if target_day is not None:
                    days_ahead = target_day - today.weekday()

                    if days_ahead <= 0:  # Target day already happened this week
                        days_ahead += 7

                    days_ahead += 7

                    next_occurrence = today + timedelta(days=days_ahead)
                    logging.info(
                        f"Calculated 'next {day_name}' as: {next_occurrence.strftime('%Y-%m-%d')}"
                    )
                    return next_occurrence.strftime("%Y-%m-%d")

        # Use dateparser for the heavy lifting
        parsed_date = dateparser.parse(
            date_string,
            settings={
                "PREFER_DAY_OF_MONTH": "first",
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
                "STRICT_PARSING": False,
                "DATE_ORDER": "MDY",  # Explicitly set MM/DD/YYYY format
            },
        )

        if parsed_date:
            return parsed_date.strftime("%Y-%m-%d")

        # Last resort: return None if we can't parse it
        logging.warning(f"Could not parse date string: {date_string}")
        return None

    except Exception as e:
        logging.error(f"Error parsing date '{date_string}': {str(e)}")
        return None


def get_date_range(period_type):
    today = datetime.now()

    if period_type == "today":
        return (today.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))

    if period_type == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return (tomorrow.strftime("%Y-%m-%d"), tomorrow.strftime("%Y-%m-%d"))

    if period_type == "this_week":
        # Start of week is Monday
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        return (start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d"))

    if period_type == "next_week":
        # Start of next week
        start_of_week = today - timedelta(days=today.weekday()) + timedelta(days=7)
        end_of_week = start_of_week + timedelta(days=6)
        return (start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d"))

    if period_type == "this_month":
        # First day of current month
        start_of_month = datetime(today.year, today.month, 1)
        # First day of next month - 1 day
        if today.month == 12:
            end_of_month = datetime(today.year + 1, 1, 1) - timedelta(days=1)
        else:
            end_of_month = datetime(today.year, today.month + 1, 1) - timedelta(days=1)
        return (start_of_month.strftime("%Y-%m-%d"), end_of_month.strftime("%Y-%m-%d"))

    if period_type == "next_month":
        if today.month == 12:
            start_of_month = datetime(today.year + 1, 1, 1)
            end_of_month = datetime(today.year + 1, 2, 1) - timedelta(days=1)
        else:
            start_of_month = datetime(today.year, today.month + 1, 1)
            if today.month == 11:
                end_of_month = datetime(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_of_month = datetime(today.year, today.month + 2, 1) - timedelta(
                    days=1
                )
        return (start_of_month.strftime("%Y-%m-%d"), end_of_month.strftime("%Y-%m-%d"))

    # Add support for specific months
    month_mapping = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    for month_name, month_num in month_mapping.items():
        if period_type.lower() == month_name:
            current_year = today.year
            # If the month has already passed this year, assume next year
            if month_num < today.month:
                current_year += 1

            start_of_month = datetime(current_year, month_num, 1)
            # Calculate the last day of the month
            if month_num == 12:
                end_of_month = datetime(current_year + 1, 1, 1) - timedelta(days=1)
            else:
                end_of_month = datetime(current_year, month_num + 1, 1) - timedelta(
                    days=1
                )
            return (
                start_of_month.strftime("%Y-%m-%d"),
                end_of_month.strftime("%Y-%m-%d"),
            )

    # Default fallback
    return (None, None)


def remove_normalized_due_date(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []

    for task in tasks:
        # Create a copy of the task dictionary
        task_copy = task.copy()
        # Remove the normalized_due_date key if it exists
        if "normalized_due_date" in task_copy:
            del task_copy["normalized_due_date"]
        result.append(task_copy)

    return result


def get_welcome_message():
    return """
==========================================================
 __  __  ____  __  __  ___  _   _  ___ 
|  \/  || ___||  \/  ||_ _|| \ | ||_ _|
| |\/| ||  _| | |\/| | | | |  \| | | | 
| |  | || |___| |  | | | | | |\  | | | 
|_|  |_||_____|_|  |_||___||_| \_||___|
                                       
==========================================================
Welcome to Memini v2.0 - Your personal task assistant
==========================================================

CAPABILITIES:
- Create tasks with due dates
- Complete, update, or delete tasks
- View tasks filtered by time
- Handle multiple requests in a single command

Try: "Add a task to submit my report by Friday" or
     "What are my pending tasks for this week?"

==========================================================

Hi, how can I help you?
"""


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    logging.info(f"Random seed set to {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Gemini with configurable parameters")

    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device number to use (default: 0)",
        required=True,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3",
        help="Model name to use (default: llama3)",
        required=True,
    )

    parser.add_argument(
        "--deterministic_dm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use deterministic dialogue manager: True or False (default: True)",
    )

    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="CRITICAL",
        help="Logging level (default: CRITICAL)",
    )

    return parser.parse_args()
