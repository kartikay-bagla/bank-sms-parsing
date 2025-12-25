#!/usr/bin/env python3
"""
Test the deployment API using test data from finetune-functiongemma.

Requires:
- The deployment server running (uvicorn deployment.main:app)
- DISABLE_ACTUAL_BUDGET=true in .env (for testing without Actual Budget)

Usage:
  # Run API tests against test data
  python -m deployment.test_api --verbose

  # Analyze database for errors and retry stats
  python -m deployment.test_api --analyze

  # Analyze a specific database file
  python -m deployment.test_api --analyze --db-path /path/to/requests.db
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import httpx


def load_test_data(path: Path) -> list[dict]:
    """Load test JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_expected(sample: dict) -> dict:
    """Extract expected tool call from sample."""
    tool_call = sample["messages"][2]["tool_calls"][0]
    return {
        "name": tool_call["function"]["name"],
        "arguments": tool_call["function"]["arguments"],
    }


def get_sms(sample: dict) -> str:
    """Extract SMS text from sample."""
    return sample["messages"][1]["content"]


async def test_single(
    client: httpx.AsyncClient,
    base_url: str,
    idx: int,
    sample: dict,
    headers: dict,
    verbose: bool,
) -> dict:
    """Test a single sample and return result."""
    sms = get_sms(sample)
    expected = get_expected(sample)

    result = {
        "idx": idx,
        "success": False,
        "correct_tool": False,
        "field_matches": {},
        "error": None,
    }

    try:
        response = await client.post(
            f"{base_url}/process-sms",
            json={"message": sms},
            headers=headers,
        )
        response.raise_for_status()
        api_result = response.json()
    except httpx.HTTPError as e:
        result["error"] = str(e)
        if verbose:
            print(f"[{idx + 1}] ERROR: {e}")
        return result

    result["success"] = True

    # Determine predicted tool
    if api_result.get("skipped_reason"):
        predicted_tool = "skip_message"
        predicted_args = {"reason": api_result["skipped_reason"]}
    elif api_result.get("transaction"):
        predicted_tool = "extract_transaction"
        predicted_args = api_result["transaction"]
    else:
        result["error"] = "Unexpected response format"
        if verbose:
            print(f"[{idx + 1}] ERROR: Unexpected response format")
        return result

    # Compare tool selection
    if predicted_tool == expected["name"]:
        result["correct_tool"] = True
        status = "✅"

        # For extract_transaction, compare fields
        if expected["name"] == "extract_transaction":
            exp_args = expected["arguments"]
            for field in ["source", "amount", "date", "destination"]:
                exp_val = exp_args.get(field)
                pred_val = predicted_args.get(field)

                # Fuzzy match for strings, exact for numbers
                if field == "amount":
                    match = float(exp_val or 0) == float(pred_val or 0)
                elif exp_val and pred_val:
                    match = (
                        str(exp_val).lower() in str(pred_val).lower()
                        or str(pred_val).lower() in str(exp_val).lower()
                    )
                else:
                    match = exp_val == pred_val

                result["field_matches"][field] = match
    else:
        status = f"❌ (expected {expected['name']}, got {predicted_tool})"

    if verbose:
        print(f"[{idx + 1}] {status} - {sms[:50]}...")

    return result


async def test_api(
    base_url: str,
    test_data: list[dict],
    api_key: str | None = None,
    parallel: int = 5,
    verbose: bool = False,
) -> dict:
    """Test the API with test data using parallel requests."""

    results = {
        "total": len(test_data),
        "correct_tool": 0,
        "wrong_tool": 0,
        "errors": 0,
        "field_matches": {"source": 0, "amount": 0, "date": 0, "destination": 0},
        "field_totals": {"source": 0, "amount": 0, "date": 0, "destination": 0},
    }

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    semaphore = asyncio.Semaphore(parallel)

    async def bounded_test(client, idx, sample):
        async with semaphore:
            return await test_single(client, base_url, idx, sample, headers, verbose)

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            bounded_test(client, idx, sample)
            for idx, sample in enumerate(test_data)
        ]
        test_results = await asyncio.gather(*tasks)

    # Aggregate results
    for r in test_results:
        if r["error"]:
            results["errors"] += 1
        elif r["correct_tool"]:
            results["correct_tool"] += 1
            for field, match in r["field_matches"].items():
                results["field_totals"][field] += 1
                if match:
                    results["field_matches"][field] += 1
        else:
            results["wrong_tool"] += 1

    return results


def analyze_db(db_path: Path, verbose: bool = False) -> dict:
    """Analyze the request database for errors and statistics."""
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "retry_distribution": defaultdict(int),
        "error_categories": defaultdict(int),
        "errors": [],
    }

    # Get totals
    cur = conn.execute(
        "SELECT COUNT(*) as total, "
        "SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count "
        "FROM request_logs"
    )
    row = cur.fetchone()
    results["total"] = row["total"]
    results["success"] = row["success_count"] or 0
    results["failed"] = results["total"] - results["success"]

    # Get retry distribution (handle missing column gracefully)
    try:
        cur = conn.execute(
            "SELECT retry_count, COUNT(*) as cnt "
            "FROM request_logs GROUP BY retry_count ORDER BY retry_count"
        )
        for row in cur.fetchall():
            results["retry_distribution"][row["retry_count"]] = row["cnt"]
    except sqlite3.OperationalError:
        # retry_count column doesn't exist in old databases
        pass

    # Get error messages
    cur = conn.execute(
        "SELECT error_message, COUNT(*) as cnt "
        "FROM request_logs WHERE success = 0 "
        "GROUP BY error_message ORDER BY cnt DESC"
    )
    for row in cur.fetchall():
        err = row["error_message"] or "Unknown error"
        results["errors"].append({"message": err, "count": row["cnt"]})

        # Categorize errors
        if "currency" in err.lower() or "INR" in err or "Rs" in err or "USD" in err:
            if "garbage" in err.lower() or "{" in err:
                results["error_categories"]["Garbage output (needs retry)"] += row["cnt"]
            else:
                results["error_categories"]["Currency in amount"] += row["cnt"]
        elif "No numeric value" in err:
            results["error_categories"]["No valid number"] += row["cnt"]
        elif "connection error" in err.lower():
            results["error_categories"]["Connection error"] += row["cnt"]
        else:
            results["error_categories"]["Other"] += row["cnt"]

    # Get sample failed requests if verbose
    if verbose and results["failed"] > 0:
        cur = conn.execute(
            "SELECT sms_message, llm_raw_response, error_message "
            "FROM request_logs WHERE success = 0 LIMIT 5"
        )
        results["sample_failures"] = [dict(row) for row in cur.fetchall()]

    conn.close()
    return results


def print_analysis_report(results: dict, verbose: bool = False):
    """Print database analysis report."""
    print("\n" + "=" * 70)
    print("DATABASE ANALYSIS REPORT")
    print("=" * 70)

    total = results["total"]
    success = results["success"]
    failed = results["failed"]

    print(f"\nTotal requests: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    if total > 0:
        print(f"Success rate: {100 * success / total:.1f}%")

    # Retry distribution
    if results["retry_distribution"]:
        print("\n" + "-" * 40)
        print("Retry Distribution:")
        for retries, count in sorted(results["retry_distribution"].items()):
            label = f"{retries} retries" if retries != 1 else "1 retry"
            if retries == 0:
                label = "No retries (first attempt)"
            print(f"  {label}: {count} requests")

        # Calculate retry effectiveness
        retried_success = sum(
            c for r, c in results["retry_distribution"].items()
            if r > 0  # Successful after retry
        )
        if retried_success > 0:
            print(f"\n  Requests recovered by retry: {retried_success}")

    # Error categories
    if results["error_categories"]:
        print("\n" + "-" * 40)
        print("Error Categories:")
        for category, count in sorted(
            results["error_categories"].items(), key=lambda x: -x[1]
        ):
            print(f"  {count:3d}x  {category}")

    # Detailed errors
    if results["errors"]:
        print("\n" + "-" * 40)
        print("Error Details (top 10):")
        for err in results["errors"][:10]:
            msg = err["message"][:60] + "..." if len(err["message"]) > 60 else err["message"]
            print(f"  {err['count']:3d}x  {msg}")

    # Sample failures
    if verbose and results.get("sample_failures"):
        print("\n" + "-" * 40)
        print("Sample Failures:")
        for i, fail in enumerate(results["sample_failures"], 1):
            print(f"\n--- Failure #{i} ---")
            sms = fail["sms_message"]
            print(f"SMS: {sms[:100]}{'...' if len(sms) > 100 else ''}")
            if fail["llm_raw_response"]:
                resp = fail["llm_raw_response"]
                print(f"LLM Response: {resp[:150]}{'...' if len(resp) > 150 else ''}")
            print(f"Error: {fail['error_message']}")


def print_report(results: dict):
    """Print test results summary."""
    print("\n" + "=" * 60)
    print("API TEST REPORT")
    print("=" * 60)

    total = results["total"]
    correct = results["correct_tool"]
    errors = results["errors"]
    tested = total - errors

    print(f"\nTotal samples: {total}")
    if errors > 0:
        print(f"Errors: {errors}")

    if tested > 0:
        pct = 100 * correct / tested
        print(f"Tool selection accuracy: {correct}/{tested} ({pct:.1f}%)")

    if results["field_totals"]["amount"] > 0:
        print("\nField accuracy (for correct extract_transaction calls):")
        for field in ["source", "amount", "date", "destination"]:
            matches = results["field_matches"][field]
            total_f = results["field_totals"][field]
            if total_f > 0:
                pct = 100 * matches / total_f
                print(f"  {field}: {matches}/{total_f} ({pct:.1f}%)")


async def main_async(args):
    """Async main function."""
    # Find test file
    if args.test_file:
        test_path = Path(args.test_file)
    else:
        # Default: look for test.jsonl in finetune data dir
        script_dir = Path(__file__).parent.parent
        test_path = script_dir / "finetune-functiongemma" / "data" / "test.jsonl"
        if not test_path.exists():
            # Try dummy file
            test_path = (
                script_dir / "finetune-functiongemma" / "data" / "dummy-test.jsonl"
            )

    if not test_path.exists():
        print(f"Error: Test file not found: {test_path}")
        print("Use --test-file to specify the path")
        sys.exit(1)

    print(f"Loading test data from: {test_path}")
    test_data = load_test_data(test_path)

    if args.max_samples:
        test_data = test_data[: args.max_samples]

    print(f"Testing {len(test_data)} samples against {args.api_url}")
    print(f"Parallel requests: {args.parallel}")
    print("=" * 60)

    # Check if server is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health = await client.get(f"{args.api_url}/health")
            health.raise_for_status()
    except httpx.HTTPError:
        print(f"\nError: Cannot connect to {args.api_url}")
        print("Make sure the server is running:")
        print("  uvicorn deployment.main:app --reload")
        sys.exit(1)

    # Run tests
    results = await test_api(
        args.api_url,
        test_data,
        api_key=args.api_key,
        parallel=args.parallel,
        verbose=args.verbose,
    )

    print_report(results)


def main():
    parser = argparse.ArgumentParser(
        description="Test deployment API with finetune test data"
    )

    # Mode selection
    parser.add_argument(
        "--analyze", "-a",
        action="store_true",
        help="Analyze database instead of running API tests",
    )
    parser.add_argument(
        "--db-path",
        default="requests.db",
        help="Path to database file (for --analyze mode)",
    )

    # API test options
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Deployment API base URL",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key if endpoint is protected",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to test",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=5,
        help="Number of parallel requests (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print details for each test/analysis",
    )

    args = parser.parse_args()

    if args.analyze:
        # Run database analysis
        db_path = Path(args.db_path)
        print(f"Analyzing database: {db_path}")
        results = analyze_db(db_path, verbose=args.verbose)
        print_analysis_report(results, verbose=args.verbose)
    else:
        # Run API tests
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
