import os
import json
import uuid
import anthropic
import fitz  # pymupdf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI(title="Research Brief API")

# ── CORS ──────────────────────────────────────────────────────────────────────
# During prototyping allow all origins.
# Before going public replace "*" with your Lovable URL:
# ["https://your-app.lovable.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

AGENT_ID       = os.environ["AGENT_ID"]
ENVIRONMENT_ID = os.environ["ENVIRONMENT_ID"]

MAX_CHARS = 20_000


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes, max_chars: int = MAX_CHARS) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    doc.close()
    return text[:max_chars]


def build_user_message(paper_text: str) -> str:
    return f"""Analyze the following research paper and return ONLY a valid JSON object — no markdown, no explanation, no code fences.

The JSON must match this exact schema:

{{
  "id": "<generate a short unique slug>",
  "opportunity_title": "<market-focused title, not the paper title>",
  "one_line_brief": "<one sentence on commercial relevance>",
  "tension_line": "<the sharpest insight in one punchy sentence, e.g. 'The signal exists. Nobody has productised it yet.'>",
  "stage": "<one of: Early Research | Prototype | Validated | Market Ready>",
  "vertical": "<primary market vertical>",
  "fit_tags": ["<tag1>", "<tag2>", "<tag3>"],
  "industry_relevance": {{ "<Industry>": <score 0-100>, "<Industry2>": <score 0-100> }},
  "scientific_summary": "<plain language, no jargon, 3-4 sentences>",
  "business_summary": "<why a startup or corporate should care, 3-4 sentences>",
  "analysis": {{
    "problem": "<what problem this solves that costs someone money>",
    "who_pays": "<specific buyers — companies, roles, sectors>",
    "gap": "<what is missing before this is a shippable product>",
    "truth_required": "<what must be true for this to become a company>"
  }},
  "audience_briefs": {{
    "startup_founder": {{
      "headline": "<punchy verdict for a founder>",
      "body": "<tactical, product-focused, 3-4 sentences>"
    }},
    "corporate": {{
      "headline": "<threat or opportunity framing>",
      "body": "<strategic risk/opportunity, 3-4 sentences>"
    }},
    "campus_resident": {{
      "headline": "<ecosystem signal framing>",
      "body": "<emerging trend, timing, why now, 3-4 sentences>"
    }}
  }},
  "recommended_action": "<one clear next step for the reader>",
  "paper_title": "<exact paper title>",
  "paper_authors": "<authors as listed>",
  "paper_venue": "<journal or conference>",
  "paper_year": <year as integer>,
  "paper_url": "#"
}}

Rules:
- Return ONLY the JSON object. No other text.
- If a field cannot be determined from the paper, use a sensible placeholder.
- industry_relevance scores are 0–100 integers. Include only industries that genuinely fit.
- fit_tags should be short (2–4 words each), concrete, and useful for filtering.
- Do not invent evidence, trials, or market readiness that isn't in the paper.

Paper to analyze:
<paper>
{paper_text}
</paper>"""


def call_agent(paper_text: str) -> dict:
    """Call the Managed Agent and return parsed JSON."""

    session = client.beta.sessions.create(
        agent=AGENT_ID,
        environment_id=ENVIRONMENT_ID,
    )

    full_response = ""

    with client.beta.sessions.events.stream(session.id) as stream:
        client.beta.sessions.events.send(
            session.id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": build_user_message(paper_text)}],
            }],
        )

        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    full_response += block.text
            elif event.type == "session.status_idle":
                break

    # Strip markdown code fences if the model added them despite instructions
    cleaned = full_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[-1]        # drop opening fence
        cleaned = cleaned.rsplit("```", 1)[0].strip() # drop closing fence
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    result = json.loads(cleaned)

    # Ensure id is always present
    if not result.get("id"):
        result["id"] = str(uuid.uuid4())[:8]

    return result


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "KnowLink Research Brief API"}


@app.post("/brief")
async def generate_brief(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Accepts a PDF upload and/or pasted text.
    Returns a JSON object matching the OpportunityBrief schema.
    """
    paper_text = ""

    if file and file.filename.lower().endswith(".pdf"):
        contents = await file.read()
        paper_text = extract_pdf_text(contents)

    if not paper_text and text:
        paper_text = text[:MAX_CHARS]

    if not paper_text:
        return {"error": "Please provide a PDF file or paste some text."}

    try:
        brief = call_agent(paper_text)
        return brief
    except json.JSONDecodeError as e:
        return {"error": f"Agent returned invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
