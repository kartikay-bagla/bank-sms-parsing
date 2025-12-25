"""
SMS to Actual Budget - FastAPI Server

Parses bank SMS messages using a fine-tuned FunctionGemma model
running on LM Studio and imports transactions to Actual Budget
via the actual-http-api wrapper.
"""

import json
import time
from contextlib import asynccontextmanager
from datetime import date

import httpx
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import SMSRequest, ExtractedTransaction, ProcessingResult
from .llm.prompts import format_prompt
from .llm.parser import (
    parse_llm_response,
    validate_transaction_args,
    InvalidLLMOutputError,
)
from .clients.actual_budget import ActualBudgetClient
from .db import init_db, get_db, RequestLog

# Retry configuration
MAX_LLM_RETRIES = 2  # Total attempts = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    # Startup: initialize database
    await init_db(settings.database_url)
    yield
    # Shutdown: nothing to cleanup (connection pool handles it)


app = FastAPI(
    title="SMS to Actual Budget",
    description="Parse bank SMS messages and import transactions to Actual Budget",
    version="1.0.0",
    lifespan=lifespan,
)

# API Key security (optional - for protecting your endpoint)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if one is configured"""
    if settings.sms_api_key and api_key != settings.sms_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


async def get_actual_client() -> ActualBudgetClient:
    """Dependency to get Actual Budget client"""
    return ActualBudgetClient(settings.actual_api_url, settings.actual_api_key)


@app.post("/process-sms", response_model=ProcessingResult)
async def process_sms(
    request: SMSRequest,
    actual_client: ActualBudgetClient = Depends(get_actual_client),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
) -> ProcessingResult:
    """
    Process an SMS message:
    1. Send to FunctionGemma for parsing
    2. If it's a transaction, import to Actual Budget
    3. Log the request to database
    4. Return the result
    """

    # Initialize log entry
    log_entry = RequestLog(
        sms_message=request.message,
        account_id=request.account_id,
    )

    # Step 1: Send SMS to LM Studio for parsing (with retry logic)
    prompt = format_prompt(request.message)
    log_entry.llm_prompt = prompt

    function_name: str | None = None
    args: dict | None = None
    total_latency = 0
    attempt = 0
    failed_responses: list[str] = []  # Track failed attempts for debugging

    for attempt in range(MAX_LLM_RETRIES + 1):
        start_time = time.time()

        # Call LM Studio
        try:
            async with httpx.AsyncClient() as client:
                llm_response = await client.post(
                    settings.lm_studio_url,
                    json={
                        "prompt": prompt,
                        "max_tokens": 200,
                        "stop": ["<end_function_call>"],
                    },
                    timeout=60.0,
                )
                llm_response.raise_for_status()
        except httpx.HTTPError as e:
            # Network errors are not retriable
            log_entry.success = False
            log_entry.error_message = f"LM Studio connection error: {str(e)}"
            db.add(log_entry)
            await db.commit()
            raise HTTPException(status_code=503, detail=log_entry.error_message)

        total_latency += int((time.time() - start_time) * 1000)

        # Parse LM Studio JSON response
        try:
            response_data = llm_response.json()
            response_text = response_data["choices"][0]["text"] + "<end_function_call>"
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            log_entry.success = False
            log_entry.error_message = f"Failed to parse LM Studio response: {str(e)}"
            log_entry.llm_raw_response = llm_response.text
            db.add(log_entry)
            await db.commit()
            raise HTTPException(status_code=502, detail=log_entry.error_message)

        # Parse the function call from LLM output
        try:
            function_name, args = parse_llm_response(response_text)

            # Validate and sanitize args for extract_transaction
            if function_name == "extract_transaction":
                args = validate_transaction_args(args)

            # Success - break out of retry loop
            log_entry.llm_raw_response = response_text
            break

        except (ValueError, InvalidLLMOutputError) as e:
            failed_responses.append(response_text[:200])  # Keep first 200 chars

            if attempt < MAX_LLM_RETRIES:
                # Retry - continue loop
                continue
            else:
                # Final attempt failed
                log_entry.success = False
                log_entry.retry_count = attempt
                log_entry.llm_raw_response = response_text
                log_entry.error_message = (
                    f"Failed to parse LLM response after {attempt + 1} attempts: {str(e)}"
                )
                db.add(log_entry)
                await db.commit()
                raise HTTPException(status_code=422, detail=log_entry.error_message)

    # Record final metrics
    log_entry.llm_latency_ms = total_latency
    log_entry.retry_count = attempt if attempt > 0 else 0
    log_entry.parsed_function = function_name

    # Step 2: Handle based on function called
    if function_name == "skip_message" and args is not None:
        log_entry.parsed_skip_reason = args.get("reason", "Unknown")
        log_entry.success = True
        db.add(log_entry)
        await db.commit()
        return ProcessingResult(
            success=True,
            message="SMS skipped - not a transaction",
            skipped_reason=log_entry.parsed_skip_reason,
        )

    if function_name == "extract_transaction" and args is not None:
        # Extract fields (amount already sanitized by validate_transaction_args)
        parsed_source = str(args.get("source", ""))
        parsed_currency = str(args.get("currency", "INR"))
        parsed_amount = float(args.get("amount", 0))  # Already validated as float
        # Use current server date instead of parsed date from SMS
        parsed_date = date.today().isoformat()
        parsed_destination = str(args.get("destination", ""))
        parsed_type = str(args.get("type", "debit"))

        # Log parsed fields
        log_entry.parsed_source = parsed_source
        log_entry.parsed_currency = parsed_currency
        log_entry.parsed_amount = parsed_amount
        log_entry.parsed_date = parsed_date
        log_entry.parsed_destination = parsed_destination
        log_entry.parsed_type = parsed_type

        # Create transaction object
        transaction = ExtractedTransaction(
            source=parsed_source,
            currency=parsed_currency,
            amount=parsed_amount,
            date=parsed_date,
            destination=parsed_destination,
            type=parsed_type,
            notes=request.message,  # Raw SMS as notes
        )

        # Skip Actual Budget import if disabled (testing mode)
        if settings.disable_actual_budget:
            log_entry.success = True
            db.add(log_entry)
            await db.commit()
            return ProcessingResult(
                success=True,
                message="Transaction parsed (Actual Budget import disabled)",
                transaction=transaction,
            )

        # Import to Actual Budget
        account_id = request.account_id or settings.default_account_id

        try:
            actual_response = await actual_client.import_transaction(
                budget_sync_id=settings.default_budget_sync_id,
                account_id=account_id,
                transaction=transaction,
            )
        except HTTPException as e:
            log_entry.success = False
            log_entry.error_message = f"Actual Budget error: {e.detail}"
            db.add(log_entry)
            await db.commit()
            raise
        except Exception as e:
            log_entry.success = False
            log_entry.error_message = f"Failed to import transaction: {str(e)}"
            db.add(log_entry)
            await db.commit()
            raise HTTPException(status_code=500, detail=log_entry.error_message)

        try:
            log_entry.actual_budget_response = json.dumps(actual_response)
        except (TypeError, ValueError):
            log_entry.actual_budget_response = str(actual_response)
        log_entry.success = True

        db.add(log_entry)
        await db.commit()
        return ProcessingResult(
            success=True,
            message="Transaction imported successfully",
            transaction=transaction,
            actual_response=actual_response,
        )

    # Unknown function
    log_entry.success = False
    log_entry.error_message = f"Unknown function returned by LLM: {function_name}"
    db.add(log_entry)
    await db.commit()
    raise HTTPException(status_code=422, detail=log_entry.error_message)


@app.get("/accounts")
async def list_accounts(
    actual_client: ActualBudgetClient = Depends(get_actual_client),
    _: str = Depends(verify_api_key),
) -> list:
    """List all accounts in the default budget"""
    try:
        return await actual_client.get_accounts(settings.default_budget_sync_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch accounts: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/logs")
async def get_logs(
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
    limit: int = 100,
    offset: int = 0,
):
    """Get recent request logs for analysis."""
    from sqlalchemy import select

    try:
        stmt = (
            select(RequestLog)
            .order_by(RequestLog.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        logs = result.scalars().all()

        return [
            {
                "id": log.id,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "sms_message": log.sms_message[:100] + "..."
                if len(log.sms_message) > 100
                else log.sms_message,
                "parsed_function": log.parsed_function,
                "parsed_amount": log.parsed_amount,
                "parsed_destination": log.parsed_destination,
                "success": log.success,
                "error_message": log.error_message,
                "llm_latency_ms": log.llm_latency_ms,
                "retry_count": log.retry_count,
            }
            for log in logs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")


@app.get("/logs/{log_id}")
async def get_log_detail(
    log_id: int,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    """Get full details of a specific log entry."""
    from sqlalchemy import select

    try:
        stmt = select(RequestLog).where(RequestLog.id == log_id)
        result = await db.execute(stmt)
        log = result.scalar_one_or_none()

        if not log:
            raise HTTPException(status_code=404, detail="Log entry not found")

        return {
            "id": log.id,
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "sms_message": log.sms_message,
            "account_id": log.account_id,
            "llm_prompt": log.llm_prompt,
            "llm_raw_response": log.llm_raw_response,
            "llm_latency_ms": log.llm_latency_ms,
            "retry_count": log.retry_count,
            "parsed_function": log.parsed_function,
            "parsed_source": log.parsed_source,
            "parsed_currency": log.parsed_currency,
            "parsed_amount": log.parsed_amount,
            "parsed_date": log.parsed_date,
            "parsed_destination": log.parsed_destination,
            "parsed_type": log.parsed_type,
            "parsed_skip_reason": log.parsed_skip_reason,
            "success": log.success,
            "error_message": log.error_message,
            "actual_budget_response": log.actual_budget_response,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch log detail: {str(e)}"
        )


# Run with: uvicorn deployment.main:app --reload

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
