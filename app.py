import json
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are an information extraction engine for an ERP system. \
Given a raw invoice/document text, extract fields EXACTLY according to these rules, \
then call the extract_invoice tool with the result. Do not include any keys other \
than the ones defined in the tool's schema.

Rules:
- vendor: the biller's proper name, exactly as written in the text.
- currency: the ISO 4217 code (USD, EUR, GBP, INR, JPY). The text may say things \
like "euros", "pounds sterling", "$", "Rs.", or "\u20b9" instead of the code directly \
- infer the correct ISO code.
- total_amount: an integer in the main unit, with no separators or symbols. The \
text may spell the number out ("twelve thousand four hundred eighty"), use standard \
grouping ("12,480"), Indian-style grouping ("1,24,800"), or a "12K" suffix - resolve \
all of these to a plain integer.
- invoice_date: normalize to YYYY-MM-DD.
- due_in_days: an integer. Resolve phrasing like "Net 30", "payable within 45 days", \
or "due in two weeks" (= 14) to the equivalent number of days.
- is_paid: a boolean inferred from wording (e.g. "paid in full" => true, \
"awaiting payment" / "outstanding" => false).
- priority: one of low, normal, high, urgent - infer from tone/wording if not \
stated explicitly.
- contact_email: lowercased.
- line_items: an array of {sku, quantity, unit_price} objects, in the order they \
appear in the text. unit_price is an integer.
- item_count: the number of line items.

Return ONLY the fields the tool schema asks for - no extra keys, no commentary."""


class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str
    text: str
    schema_: Dict[str, Any] = Field(alias="schema")


def call_claude(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not configured on the server.",
        )

    tool = {
        "name": "extract_invoice",
        "description": "Return the extracted invoice/document fields exactly as specified.",
        "input_schema": schema,
    }

    payload = {
        "model": MODEL,
        "max_tokens": 2000,
        # Forced tool_choice is incompatible with Sonnet 5's default adaptive
        # thinking, so thinking must be explicitly disabled here.
        "thinking": {"type": "disabled"},
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": "Extract the fields from this document:\n\n" + text,
            }
        ],
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": "extract_invoice"},
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    try:
        resp = httpx.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=55.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Anthropic API error: {e.response.status_code} {e.response.text}",
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API request failed: {e}")

    data = resp.json()

    for block in data.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "extract_invoice":
            return block["input"]

    raise HTTPException(status_code=502, detail="Model did not return a tool call.")


@app.post("/extract")
def extract(req: ExtractRequest):
    return call_claude(req.text, req.schema_)
