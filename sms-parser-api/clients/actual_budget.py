"""
Client for interacting with Actual Budget HTTP API.
"""

from datetime import datetime
import httpx
from fastapi import HTTPException

from ..models import ExtractedTransaction


class ActualBudgetClient:
    """Client for interacting with Actual Budget HTTP API"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    async def import_transaction(
        self, budget_sync_id: str, account_id: str, transaction: ExtractedTransaction
    ) -> dict:
        """
        Import a transaction to Actual Budget.

        The actual-http-api exposes endpoints like:
        POST /v1/budgets/{syncId}/accounts/{accountId}/transactions/import
        """
        # Convert amount to integer cents (Actual uses amount * 100)
        # Negative for debits (money out), positive for credits (money in)
        amount_cents = int(transaction.amount * 100)
        if transaction.type.lower() == "debit":
            amount_cents = -amount_cents

        # Parse and format the date
        date_str = self.normalize_date(transaction.date)

        # Build the transaction payload
        payload = {
            "transactions": [
                {
                    "date": date_str,
                    "amount": amount_cents,
                    "payee_name": transaction.destination,
                    "imported_payee": transaction.destination,
                    "notes": transaction.notes or "",
                    # Generate a unique imported_id to prevent duplicates
                    "imported_id": f"sms_{date_str}_{abs(amount_cents)}_{transaction.destination[:20]}",
                }
            ]
        }

        url = f"{self.base_url}/v1/budgets/{budget_sync_id}/accounts/{account_id}/transactions/import"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, headers=self.headers, json=payload, timeout=30.0
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Actual API error: {response.text}",
                )

            return response.json()

    def normalize_date(self, date_str: str) -> str:
        """
        Normalize various date formats to YYYY-MM-DD.

        Handles formats like:
        - 01-09-23 (DD-MM-YY)
        - 23-Sep-23 (DD-Mon-YY)
        - 2023-09-23 (YYYY-MM-DD)
        """
        date_str = date_str.strip().rstrip("s")  # Remove trailing 's' if present

        # Try different formats
        formats = [
            "%Y-%m-%d",  # 2023-09-23
            "%d-%m-%y",  # 01-09-23
            "%d-%b-%y",  # 23-Sep-23
            "%d-%m-%Y",  # 01-09-2023
            "%d-%b-%Y",  # 23-Sep-2023
            "%d/%m/%y",  # 01/09/23
            "%d/%m/%Y",  # 01/09/2023
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If nothing works, return as-is and hope for the best
        return date_str

    async def get_accounts(self, budget_sync_id: str) -> list:
        """Get all accounts for a budget"""
        url = f"{self.base_url}/v1/budgets/{budget_sync_id}/accounts"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, timeout=30.0)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Actual API error: {response.text}",
                )

            return response.json()
