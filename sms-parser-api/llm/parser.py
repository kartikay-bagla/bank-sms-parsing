"""
LLM response parsing utilities for FunctionGemma output.
"""

import re


class InvalidLLMOutputError(Exception):
    """Raised when LLM output is malformed and should trigger a retry."""

    pass


def sanitize_amount(value: str) -> float:
    """
    Extract numeric amount from potentially malformed string.

    Handles common LLM mistakes:
    - Currency prefixes: "INR 194.00" -> 194.0
    - Currency symbols: "Rs.420.0" -> 420.0
    - Formatted numbers: "1,00,000.50" -> 100000.5
    - Garbage/nested objects: raises InvalidLLMOutputError

    Returns:
        float: The extracted amount

    Raises:
        InvalidLLMOutputError: If amount contains garbage (nested braces, etc.)
        ValueError: If no valid number can be extracted
    """
    original = value
    value = value.strip()

    # Detect garbage output (nested braces, function-like patterns)
    if "{" in value or "(" in value or value.count(":") > 0:
        raise InvalidLLMOutputError(
            f"Amount contains garbage/nested structure: {original}"
        )

    # Remove common currency prefixes (case-insensitive)
    currency_patterns = [
        r"^INR\s*",
        r"^Rs\.?\s*",
        r"^USD\s*",
        r"^EUR\s*",
        r"^₹\s*",
        r"^\$\s*",
    ]
    for pattern in currency_patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)

    # Remove commas (handles Indian 1,00,000 and Western 100,000 formats)
    value = value.replace(",", "")

    # Remove any trailing non-numeric characters (e.g., "194.00`" -> "194.00")
    value = re.sub(r"[^\d.]+$", "", value)
    value = re.sub(r"^[^\d.]+", "", value)

    # Try to extract a number
    value = value.strip()
    if not value:
        raise ValueError(f"No numeric value found in: {original}")

    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Could not convert to float: {original}")


def parse_llm_response(response_text: str) -> tuple[str, dict]:
    """
    Parse the FunctionGemma function call response.

    Returns:
        tuple of (function_name, arguments_dict)
    """
    # Expected format: <start_function_call>call:function_name{...}<end_function_call>
    # Extract function name and arguments

    # Remove the end tag if present
    response_text = response_text.replace("<end_function_call>", "").strip()

    # Match the function call pattern
    match = re.match(
        r"<start_function_call>call:(\w+)\{(.+)\}", response_text, re.DOTALL
    )
    if not match:
        raise ValueError(f"Could not parse LLM response: {response_text}")

    function_name = match.group(1)
    args_str = match.group(2)

    # Parse arguments in custom format (not JSON)
    # Example: amount:7113.0,date:<escape>2023-09-23<escape>,destination:<escape>YATRA<escape>
    args = {}

    # Split by comma, but be careful with escaped strings
    current_key = None
    current_value = ""
    in_escape = False

    i = 0
    while i < len(args_str):
        char = args_str[i]

        if args_str[i : i + 8] == "<escape>":
            if in_escape:
                # End of escaped string
                in_escape = False
                i += 8
                continue
            else:
                # Start of escaped string
                in_escape = True
                i += 8
                continue

        if in_escape:
            current_value += char
            i += 1
            continue

        if char == ":" and current_key is None:
            current_key = current_value
            current_value = ""
            i += 1
            continue

        if char == "," and current_key is not None:
            # Save current key-value pair
            args[current_key] = parse_value(current_value)
            current_key = None
            current_value = ""
            i += 1
            continue

        current_value += char
        i += 1

    # Don't forget the last key-value pair
    if current_key is not None:
        args[current_key] = parse_value(current_value)

    return function_name, args


def parse_value(value: str) -> str | float | int:
    """Parse a value string into appropriate type"""
    value = value.strip()

    # Try to parse as number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def validate_transaction_args(args: dict) -> dict:
    """
    Validate and sanitize transaction arguments.

    This function:
    1. Validates required fields exist
    2. Sanitizes the amount field (handles currency prefixes, garbage)
    3. Returns sanitized args dict

    Raises:
        InvalidLLMOutputError: If args contain garbage that should trigger retry
        ValueError: If required fields missing or amount invalid
    """
    required_fields = ["amount"]  # At minimum we need amount

    for field in required_fields:
        if field not in args:
            raise ValueError(f"Missing required field: {field}")

    # Sanitize amount field
    amount_value = args.get("amount")
    if amount_value is not None:
        if isinstance(amount_value, (int, float)):
            args["amount"] = float(amount_value)
        else:
            # String amount - needs sanitization
            args["amount"] = sanitize_amount(str(amount_value))

    return args
