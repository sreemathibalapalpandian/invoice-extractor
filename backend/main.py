import os, io, uuid, asyncio, json, re
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pandas as pd
from fpdf import FPDF
from groq import AsyncGroq

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "❌ GROQ_API_KEY missing. Add it to backend/.env or Render dashboard."
    )

CACHE = {}
CACHE_TTL = 600


def extract_text(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = " ".join(page.extract_text() or "" for page in pdf.pages)
            return text[:8000]
    except:
        return ""


def regex_fallback(text: str, field: str) -> str:
    if "date" in field.lower():
        m = re.search(
            r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b",
            text,
        )
        return m.group(0) if m else "-"
    if any(k in field.lower() for k in ["amount", "total", "paid", "price"]):
        m = re.search(r"\$?[\d,]+\.?\d*", text)
        return "$" + m.group(0).lstrip("$") if m else "-"
    return "-"


async def call_groq(client: AsyncGroq, prompt: str, retries: int = 3):
    for attempt in range(retries):
        try:
            res = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = res.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt < retries - 1:
                prompt += (
                    "\n\nIMPORTANT: Return ONLY valid JSON. No markdown, no extra text."
                )
                await asyncio.sleep(1)
                continue
            raise
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                await asyncio.sleep(2**attempt)
                continue
            raise
    raise Exception("AI failed after retries")


@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...), columns: str = Query(...)):
    headers = [h.strip().rstrip(".").strip() for h in columns.split(",") if h.strip()]
    if not headers or not files:
        raise HTTPException(400, "Missing columns or files")

    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(3)

    async def process_file(f):
        async with semaphore:
            content = await f.read()
            text = await asyncio.to_thread(extract_text, content)
            if not text.strip():
                return {"Source_File": f.filename, **{h: "-" for h in headers}}

            prompt = f"""Extract ONLY these exact fields from the invoice text. Return strict JSON. Use null if truly missing.
Fields: {json.dumps(headers)}

Rules:
- company name -> Top vendor/business name. Ignore addresses.
- date paid -> Look for "Paid Date" or "Payment Date". If missing, use "Invoice Date" or "Due Date". Keep exact format.
- invoice amount paid -> FINAL grand total/amount due. Ignore line items, tax, shipping, subtotals. Include $/currency.
- If a field isn't labeled exactly, match by context. Never guess random numbers.
- Return ONLY valid JSON with the exact keys above.

Text: {text}"""

            try:
                data = await call_groq(client, prompt)
                row = {"Source_File": f.filename}
                for h in headers:
                    val = data.get(h)
                    if val in (None, "", "-"):
                        val = regex_fallback(text, h)
                    row[h] = str(val).strip() if val else "-"
                return row
            except Exception as e:
                print(f"❌ AI failed for {f.filename}: {e}")
                return {"Source_File": f.filename, **{h: "-" for h in headers}}

    results = await asyncio.gather(*(process_file(f) for f in files))
    df = pd.DataFrame(results)
    df = df[["Source_File"] + headers]

    task_id = str(uuid.uuid4())
    CACHE[task_id] = {
        "df": df,
        "expires": datetime.now() + timedelta(seconds=CACHE_TTL),
    }

    return {
        "task_id": task_id,
        "preview": df.head(50).to_dict(orient="records"),
        "total": len(df),
    }


@app.get("/api/download/excel/{task_id}")
async def download_excel(task_id: str):
    if task_id not in CACHE:
        raise HTTPException(404, "Session expired")
    buf = io.BytesIO()
    CACHE[task_id]["df"].to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=invoices.xlsx"},
    )


@app.get("/api/download/pdf/{task_id}")
async def download_pdf(task_id: str):
    if task_id not in CACHE:
        raise HTTPException(404, "Session expired")
    try:
        df = CACHE[task_id]["df"]
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()
        pdf.set_font("Helvetica", size=8)

        cols = df.columns.tolist()
        col_width = max(20, min(45, 190 // max(len(cols), 1)))

        # Header
        pdf.set_fill_color(240, 245, 255)
        for col in cols:
            safe = str(col)[:22].encode("ascii", errors="replace").decode("ascii")
            pdf.cell(col_width, 7, safe, border=1, fill=True)
        pdf.ln()

        # Rows
        for row in df.itertuples(index=False):
            for val in row:
                safe = str(val)[:25].encode("ascii", errors="replace").decode("ascii")
                pdf.cell(col_width, 6, safe, border=1)
            pdf.ln()

        # ✅ CRITICAL FIX: dest="B" returns raw bytes. No encoding step needed.
        pdf_bytes = pdf.output(dest="B")
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=invoices.pdf"},
        )
    except Exception as e:
        import traceback

        print("❌ PDF GENERATION FAILED:")
        traceback.print_exc()
        raise HTTPException(500, f"PDF generation error: {str(e)}")
