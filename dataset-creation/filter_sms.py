import json
import pandas as pd


def load_messages(filepath: str) -> pd.DataFrame:
    """Load SMS messages from JSON file and normalize to DataFrame."""
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    df = df[["body", "date"]]
    df.insert(0, "correct", "y")
    return df


def filter_transaction_messages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter messages into transaction and non-transaction categories.

    Transaction messages must:
    1. Contain at least one transaction-related keyword
    2. Contain at least 2 numbers
    """
    filters = "debit credit spent payment transfer sent received paid inr usd eur gbp".split()

    # Filter messages related to transactions
    transaction_msgs = df[df["body"].str.contains("|".join(filters), case=False, na=False)]

    # Transaction messages must contain at least 2 numbers
    transaction_msgs = transaction_msgs[
        transaction_msgs["body"].str.count(r"\d+") >= 2
    ]

    # filter messages with "OTP" in them
    transaction_msgs = transaction_msgs[
        ~transaction_msgs["body"].str.contains("OTP", case=False, na=False)
    ]

    non_transaction_msgs = df.drop(transaction_msgs.index)

    return transaction_msgs, non_transaction_msgs


def main():
    from pathlib import Path

    script_dir = Path(__file__).parent
    input_file = script_dir / "source-sms/Messages_2025_01_24_18_59_34.json"
    output_dir = script_dir / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_messages(str(input_file))
    transaction_msgs, non_transaction_msgs = filter_transaction_messages(df)

    # Save both dataframes to CSV
    transaction_msgs.to_csv(output_dir / "transaction_messages.csv", index=False)
    non_transaction_msgs.to_csv(output_dir / "non_transaction_messages.csv", index=False)

    print(f"Total messages: {len(df)}")
    print(f"Transaction messages: {len(transaction_msgs)}")
    print(f"Non-transaction messages: {len(non_transaction_msgs)}")


if __name__ == "__main__":
    main()
