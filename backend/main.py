import os, io, uuid, asyncio, json, re, traceback
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
    raise RuntimeError("❌ GROQ_API_KEY missing.")

CACHE = {}
CACHE_TTL = 600

def extract_text(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract ALL pages, but prioritize LAST 3000 chars (where totals live)
            full = " ".join(page.extract_text() or "" for page in pdf.pages)
            return full[-4000:] if len(full) > 4000 else full  # Keep tail + some context
    except: return ""

def extract_final_amount(text: str) -> str:
    """Bulletproof final amount extractor: keyword-anchored + value-validated"""
    # 1. Look for explicit total keywords at END of text (most reliable)
    total_patterns = [
        r'(?:grand\s*total|total\s*due|amount\s*due|balance\s*due|final\s*total|total\s*amount)\s*:?\s*\$?([\d,]+\.?\d*)',
        r'\bTOTAL\s*:?\s*\$?([\d,]+\.?\d*)',
        r'(?:net\s*total|sum\s*total)\s*:?\s*\$?([\d,]+\.?\d*)'
    ]
    for pat in total_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(',', '')
            try:
                return f"${float(val):,.2f}"
            except: return f"${val}"
    
    # 2. Fallback: grab ALL $ amounts, return the LARGEST (usually the total)
    amounts = re.findall(r'\$?([\d,]+\.?\d{2})\b', text)
    if amounts:
        clean = [float(a.replace(',', '')) for a in amounts if re.match(r'^[\d,]+\.?\d{2}$', a.replace(',', ''))]
        if clean:
            return f"${max(clean):,.2f}"
    
    return "-"

def extract_date(text: str) -> str:
    """Smart date extractor with fallbacks"""
    # 1. Look for explicit date keywords
    date_pats = [
        r'(?:paid\s*date|payment\s*date|date\s*paid)\s*:?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})',
        r'(?:invoice\s*date|inv\s*date)\s*:?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})',
        r'(?:due\s*date)\s*:?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})'
    ]
    for pat in date_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    
    # 2. Fallback: first standalone date-like pattern
    m = re.search(r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b', text)
    return m.group(1).strip() if m else "-"

def extract_company(text: str) -> str:
    """Company = first meaningful non-address line"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:  # Check top 8 lines
        if len(line) > 5 and line[0].isalpha() and not re.search(r'(?:street|ave|road|blvd|suite|\d{5,}|tx|ny|ca|il|wa)', line, re.I):
            return line.split(',')[0].strip()  # Cut off address if present
    return "-"

async def call_groq(client: AsyncGroq, prompt: str, retries: int = 3):
    for attempt in range(retries):
        try:
            res = await client.chat.completions.create(
                model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=300, response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.IGNORECASE))
        except json.JSONDecodeError:
            if attempt < retries - 1:
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON."
                await asyncio.sleep(1); continue
            raise
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                await asyncio.sleep(2 ** attempt); continue
            raise
    raise Exception("AI failed")

@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...), columns: str = Query(...)):
    headers = [h.strip().rstrip('.').strip() for h in columns.split(",") if h.strip()]
    if not headers or not files: raise HTTPException(400, "Missing")
    client = AsyncGroq(api_key=GROQ_API_KEY)
    semaphore = asyncio.Semaphore(3)

    async def process(f):
        async with semaphore:
            content = await f.read()
            text = await asyncio.to_thread(extract_text, content)
            if not text.strip(): return {"Source_File": f.filename, **{h: "-" for h in headers}}
            
            # 🔹 SMART PROMPT: Explicit rules for each field type
            prompt = f"""Extract ONLY these exact fields. Return strict JSON. Use null if truly missing.
Fields: {json.dumps(headers)}

STRICT RULES:
- For ANY field containing "amount", "total", "paid", or "price": 
  → Extract ONLY the FINAL grand total (e.g., "$1,325.00"). 
  → IGNORE line items, shipping, tax, subtotals. 
  → Look for keywords: "TOTAL:", "Grand Total", "Amount Due".
- For ANY field containing "date": 
  → Extract invoice/paid/due date. Keep original format.
- For ANY field containing "company", "vendor", or "name": 
  → Extract top business name. Ignore addresses.
- Return ONLY valid JSON with exact keys. No extra text.

Text excerpt (tail-focused): {text}"""
            
            try:
                data = await call_groq(client, prompt)
                row = {"Source_File": f.filename}
                for h in headers:
                    val = data.get(h)
                    h_lower = h.lower()
                    
                    # 🔹 Field-specific validation & fallback
                    if any(k in h_lower for k in ["amount", "total", "paid", "price"]):
                        # Validate AI amount: must look like $X,XXX.XX
                        if val and re.match(r'^\$?[\d,]+\.?\d*$', str(val)):
                            row[h] = str(val).strip()
                        else:
                            row[h] = await asyncio.to_thread(extract_final_amount, text)
                            print(f"🔁 Fallback amount for {f.filename}: {row[h]}")
                    elif "date" in h_lower:
                        row[h] = val if val and re.match(r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', str(val)) else await asyncio.to_thread(extract_date, text)
                    elif any(k in h_lower for k in ["company", "vendor", "name"]):
                        row[h] = val if val and len(str(val)) > 3 else await asyncio.to_thread(extract_company, text)
                    else:
                        row[h] = str(val).strip() if val and str(val).strip() else "-"
                return row
            except Exception as e:
                print(f"❌ AI failed for {f.filename}: {e}")
                # 🔹 Full fallback using deterministic rules
                row = {"Source_File": f.filename}
                for h in headers:
                    h_lower = h.lower()
                    if any(k in h_lower for k in ["amount", "total", "paid", "price"]):
                        row[h] = await asyncio.to_thread(extract_final_amount, text)
                    elif "date" in h_lower:
                        row[h] = await asyncio.to_thread(extract_date, text)
                    elif any(k in h_lower for k in ["company", "vendor", "name"]):
                        row[h] = await asyncio.to_thread(extract_company, text)
                    else:
                        row[h] = "-"
                return row

    results = await asyncio.gather(*(process(f) for f in files))
    df = pd.DataFrame(results)[["Source_File"] + headers]
    tid = str(uuid.uuid4())
    CACHE[tid] = {"df": df, "expires": datetime.now() + timedelta(seconds=CACHE_TTL)}
    return {"task_id": tid, "preview": df.head(50).to_dict(orient="records"), "total": len(df)}

@app.get("/api/download/excel/{task_id}")
async def download_excel(task_id: str):
    if task_id not in CACHE: raise HTTPException(404, "Expired")
    buf = io.BytesIO()
    CACHE[task_id]["df"].to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=invoices.xlsx"})

@app.get("/api/download/pdf/{task_id}")
async def download_pdf(task_id: str):
    if task_id not in CACHE: raise HTTPException(404, "Expired")
    try:
        df = CACHE[task_id]["df"]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=8)
        cols = df.columns.tolist()
        col_w = max(30, min(50, 190 // max(len(cols), 1)))
        def safe(txt, limit=18):
            if not txt: return ""
            txt = str(txt).replace("\n", " ").replace("\r", "")
            return txt[:limit] + (".." if len(txt) > limit else "")
        pdf.set_fill_color(230, 241, 255)
        for c in cols: pdf.cell(col_w, 7, safe(c, 20), border=1, fill=True, align="L")
        pdf.ln()
        for row in df.itertuples(index=False):
            for v in row: pdf.cell(col_w, 6, safe(v, 18), border=1, align="L")
            pdf.ln()
        out = pdf.output(dest="S")
        pdf_bytes = out.encode("latin-1", errors="replace") if isinstance(out, str) else out
        return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf",
                                 headers={"Content-Disposition": "attachment; filename=invoices.pdf"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"PDF error: {str(e)}")