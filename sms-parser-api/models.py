"""
Pydantic models for request/response handling.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SMSRequest(BaseModel):
    """Incoming SMS message to be processed"""

    message: str
    account_id: Optional[str] = Field(
        default=None, description="Override default account ID for this transaction"
    )


class ExtractedTransaction(BaseModel):
    """Transaction details extracted from SMS"""

    source: str
    currency: str
    amount: float
    date: str
    destination: str
    type: str  # 'debit' or 'credit'
    notes: Optional[str] = None  # Raw SMS message


class SkippedMessage(BaseModel):
    """Reason for skipping a non-transaction SMS"""

    reason: str


class ProcessingResult(BaseModel):
    """Result of processing an SMS"""

    success: bool
    message: str
    transaction: Optional[ExtractedTransaction] = None
    actual_response: Optional[dict] = None
    skipped_reason: Optional[str] = None
