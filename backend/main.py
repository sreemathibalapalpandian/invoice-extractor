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
    raise RuntimeError("❌ GROQ_API_KEY missing. Add it to backend/.env or Render dashboard.")

CACHE = {}
CACHE_TTL = 600

def extract_text(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)[:8000]
    except:
        return ""

def regex_fallback(text: str, field: str) -> str:
    if "date" in field.lower():
        m = re.search(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b', text)
        return m.group(0) if m else "-"
    if any(k in field.lower() for k in ["amount", "total", "paid", "price"]):
        m = re.search(r'\$?[\d,]+\.?\d*', text)
        return "$" + m.group(0).lstrip("$") if m else "-"
    return "-"

async def call_groq(client: AsyncGroq, prompt: str, retries: int = 3):
    for attempt in range(retries):
        try:
            res = await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=300, response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content.strip()
            return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.IGNORECASE))
        except json.JSONDecodeError:
            if attempt < retries - 1:
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No markdown."
                await asyncio.sleep(1)
                continue
            raise
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise Exception("AI failed after retries")

@app.post("/api/extract")
async def extract(files: list[UploadFile] = File(...), columns: str = Query(...)):
    headers = [h.strip().rstrip('.').strip() for h in columns.split(",") if h.strip()]
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

            prompt = f"""Extract ONLY these exact fields. Return strict JSON. Use null if missing.
Fields: {json.dumps(headers)}
Rules:
- company name -> Top vendor/business name.
- date paid -> Paid/Invoice/Due date. Keep original format.
- invoice amount paid -> FINAL grand total only.
- Return ONLY valid JSON with exact keys.
Text: {text}"""
            
            try:
                data = await call_groq(client, prompt)
                row = {"Source_File": f.filename}
                for h in headers:
                    val = data.get(h)
                    if val in (None, "", "-"): val = regex_fallback(text, h)
                    row[h] = str(val).strip() if val else "-"
                return row
            except Exception as e:
                print(f"❌ AI failed for {f.filename}: {e}")
                return {"Source_File": f.filename, **{h: "-" for h in headers}}

    results = await asyncio.gather(*(process_file(f) for f in files))
    df = pd.DataFrame(results)
    df = df[["Source_File"] + headers]

    task_id = str(uuid.uuid4())
    CACHE[task_id] = {"df": df, "expires": datetime.now() + timedelta(seconds=CACHE_TTL)}
    
    return {"task_id": task_id, "preview": df.head(50).to_dict(orient="records"), "total": len(df)}

@app.get("/api/download/excel/{task_id}")
async def download_excel(task_id: str):
    if task_id not in CACHE: raise HTTPException(404, "Session expired")
    buf = io.BytesIO()
    CACHE[task_id]["df"].to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=invoices.xlsx"})

@app.get("/api/download/pdf/{task_id}")
async def download_pdf(task_id: str):
    if task_id not in CACHE: raise HTTPException(404, "Session expired")
    try:
        df = CACHE[task_id]["df"]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=8)
        
        cols = df.columns.tolist()
        col_w = max(25, min(45, 190 // max(len(cols), 1)))
        
        pdf.set_fill_color(230, 241, 255)
        for col in cols:
            pdf.cell(col_w, 7, str(col)[:20], border=1, fill=True)
        pdf.ln()
        
        for row in df.itertuples(index=False):
            for val in row:
                pdf.cell(col_w, 6, str(val if val is not None else "")[:25], border=1)
            pdf.ln()
            
        # ✅ CRITICAL: fpdf2 output returns string. Encode to latin-1 (standard PDF encoding)
        pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="replace")
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=invoices.pdf"}
        )
    except Exception as e:
        print("🔴 PDF GENERATION FAILED:")
        traceback.print_exc()
        raise HTTPException(500, f"PDF error: {str(e)}")