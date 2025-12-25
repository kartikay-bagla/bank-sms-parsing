"""SQLAlchemy models for request logging."""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class RequestLog(Base):
    """Log of all SMS processing requests for analysis and model improvement."""

    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Input
    sms_message: Mapped[str] = mapped_column(Text, nullable=False)
    account_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # LLM interaction
    llm_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_raw_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_latency_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    retry_count: Mapped[int] = mapped_column(default=0)  # Number of LLM retries

    # Parsed output
    parsed_function: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # extract_transaction or skip_message
    parsed_source: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    parsed_currency: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    parsed_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    parsed_date: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    parsed_destination: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True
    )
    parsed_type: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )  # debit or credit
    parsed_skip_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Result
    success: Mapped[bool] = mapped_column(default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    actual_budget_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<RequestLog(id={self.id}, function={self.parsed_function})>"
