"""
HW11 Agent – SEC Financial Analyst
-----------------------------------
An agentic loop that answers financial questions about publicly traded companies
using three retrieval tools:

  1. search_sec_kb        – cosine-similarity search over the local vector store
                            built by rohanvinayaksagvekar_build_kb.py (10-K filings).
  2. get_company_facts    – fetches structured XBRL financials (revenue, R&D, capex,
                            net income) from the SEC EDGAR company-facts API.
  3. get_recent_filings   – fetches the most recent filing history (10-K / 10-Q / 8-K)
                            for a ticker from the SEC EDGAR submissions API.

Implementation notes:
  - Uses OpenAI function calling (chat.completions.create with tools=[]).
  - The agentic loop runs until the model returns finish_reason "stop".
  - All SEC HTTP calls go through the EDGAR public APIs (no auth required).
  - Make sure kb_out/ exists by running the build script first.

Example query: "Compare R&D spending and capex for Apple and Microsoft in their
               most recent fiscal year, and list any recent 8-K filings for both."
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
KB_DIR = HERE / "kb_out"

load_dotenv(HERE / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
    "META": "0001326801",
    "TSLA": "0001318605",
    "NFLX": "0001065280",
}

EMBED_MODEL = "text-embedding-3-small"
INFER_MODEL = "gpt-4o-mini"
TOP_K       = 15
HEADERS     = {"User-Agent": "Rohan Sagvekar rsagveka@stevens.edu"}

# Tool 1 – Local vector store search (cosine similarity over 10-K embeddings)

def search_sec_kb(query, tickers=None, k=TOP_K):
    embeddings = np.load(KB_DIR / "embeddings.npy")
    with open(KB_DIR / "metadata.json") as f:
        metadata = json.load(f)

    res = client.embeddings.create(input=[query], model=EMBED_MODEL, encoding_format="float")
    q_vec = np.array(res.data[0].embedding, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    scores = embeddings @ q_vec / np.where(norms == 0, 1, norms)

    if tickers:
        results = []
        for ticker in tickers:
            idx = [i for i, m in enumerate(metadata) if m.get("ticker") == ticker.upper()]
            if not idx:
                continue
            top = np.argsort(scores[idx])[-k:][::-1]
            results.extend(metadata[idx[i]]["text"] for i in top)
    else:
        top_idx = np.argsort(scores)[-k:][::-1]
        results = [metadata[i]["text"] for i in top_idx]

    return "\n\n---\n\n".join(results) if results else "No results found."

# Tool 2 – SEC EDGAR company facts (XBRL structured financials)

def get_company_facts(ticker):
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return f"Ticker {ticker} not in the supported list."

    time.sleep(0.11)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    facts = resp.json()

    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    keys_of_interest = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "NetIncomeLoss",
        "ResearchAndDevelopmentExpense",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "OperatingIncomeLoss",
        "EarningsPerShareBasic",
    ]

    summary = {}
    for key in keys_of_interest:
        if key not in us_gaap:
            continue
        usd_vals = us_gaap[key].get("units", {}).get("USD", [])
        annual = [v for v in usd_vals if v.get("form") == "10-K"]
        if annual:
            latest = sorted(annual, key=lambda x: x["end"])[-1]
            summary[key] = {"value_usd": latest["val"], "period_end": latest["end"]}

    return json.dumps({ticker.upper(): summary}, indent=2) if summary else f"No XBRL data found for {ticker}."

# Tool 3 – SEC EDGAR submissions (recent filing list)

def get_recent_filings(ticker, form_types=None):
    cik = TICKER_TO_CIK.get(ticker.upper())
    if not cik:
        return f"Ticker {ticker} not in the supported list."

    time.sleep(0.11)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    allowed = set(form_types) if form_types else {"10-K", "10-Q", "8-K"}
    recent = data["filings"]["recent"]
    filings = []
    for form, date, acc, desc in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent.get("primaryDocDescription", [""] * len(recent["form"])),
    ):
        if form in allowed:
            filings.append({"form": form, "date": date, "accession": acc, "description": desc})
        if len(filings) >= 15:
            break

    return json.dumps({ticker.upper(): filings}, indent=2) if filings else f"No filings found for {ticker}."

# Tool schema for the OpenAI API

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_sec_kb",
            "description": (
                "Search the local vector store of SEC 10-K filings using semantic "
                "similarity. Returns the most relevant text passages from annual reports. "
                "Use this to answer qualitative questions or find specific disclosures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tickers to restrict the search (e.g. ['AAPL', 'MSFT']).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_facts",
            "description": (
                "Fetch structured XBRL financial data for a company directly from the "
                "SEC EDGAR company-facts API. Returns key annual metrics such as revenue, "
                "net income, R&D expense, and capex from the most recent 10-K."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL').",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_filings",
            "description": (
                "Retrieve the most recent SEC filing history for a company from the "
                "EDGAR submissions API. Returns form type, filing date, and accession "
                "number for the latest 10-K, 10-Q, and 8-K filings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'AAPL').",
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of form types to filter (e.g. ['10-K', '8-K']). Defaults to 10-K, 10-Q, 8-K.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


# Agentic loop

SYSTEM = (
    "You are a senior financial analyst with access to SEC EDGAR data. "
    "You MUST follow this research process before writing any final answer:\n"
    "  1. Call search_sec_kb (with company-specific tickers) to gather qualitative "
    "disclosures from the 10-K filings. Search multiple times with different queries "
    "if the question has several sub-topics (e.g. one search for R&D, another for capex).\n"
    "  2. Call get_company_facts for every company mentioned to obtain exact financial figures.\n"
    "  3. Call get_recent_filings for every company mentioned to get filing history.\n"
    "Only write your final answer after completing all tool calls. "
    "In your final answer: include ALL filings returned by get_recent_filings (do not truncate the list), "
    "report every financial metric returned by get_company_facts, "
    "and cite specific dollar figures and filing dates throughout."
)

QUERY = (
    "Using the 10-K filings, summarize how Apple and Microsoft describe their "
    "R&D strategy and capital expenditure priorities, compare the exact dollar "
    "figures for both, and list any recent 8-K filings for each company."
)

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": QUERY},
    ]

    print(f"\n[query] {QUERY}\n")

    while True:
        response = client.chat.completions.create(
            model=INFER_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg    = choice.message

        messages.append(msg)

        if choice.finish_reason == "stop":
            print("\n[answer]\n")
            print(msg.content)
            break

        for tc in msg.tool_calls or []:
            args = json.loads(tc.function.arguments)
            print(f"[tool]  {tc.function.name}({args})")

            if tc.function.name == "search_sec_kb":
                result = search_sec_kb(args["query"], args.get("tickers"))
            elif tc.function.name == "get_company_facts":
                result = get_company_facts(args["ticker"])
            elif tc.function.name == "get_recent_filings":
                result = get_recent_filings(args["ticker"], args.get("form_types"))
            else:
                result = f"Unknown tool: {tc.function.name}"

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })