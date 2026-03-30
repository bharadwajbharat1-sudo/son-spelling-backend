import json
import os
import random
from pathlib import Path
from typing import List

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Initialize FastAPI
app = FastAPI(title="Kid Practice Backend", version="1.0.0")

# 2. Consolidated Middleware (ONE call only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Configure Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    # This will show up in your Render logs if the key is missing
    print("WARNING: GOOGLE_API_KEY is not set in environment variables.")
else:
    genai.configure(api_key=google_api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

# --- Mistake Tracking Logic ---
BASE_DIR = Path(__file__).resolve().parent
MISTAKES_FILE = BASE_DIR / "mistakes.json"

def _ensure_mistakes_file() -> None:
    if not MISTAKES_FILE.exists():
        MISTAKES_FILE.write_text("[]", encoding="utf-8")

def load_mistakes() -> List[str]:
    _ensure_mistakes_file()
    try:
        raw = json.loads(MISTAKES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        raw = []
    return [str(word).strip().lower() for word in raw if str(word).strip()]

def save_mistakes(words: List[str]) -> None:
    normalized = sorted({w.strip().lower() for w in words if w and w.strip()})
    MISTAKES_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

# --- API Routes ---

class MistakeInput(BaseModel):
    word: str

@app.get("/health")
async def health_check():
    return {"status": "ok", "provider": "google-gemini"}

@app.get("/generate")
async def generate_dynamic_content(mode: str = "sentence", level: int = 1, topic: str = "anything"):
    # Updated Prompt: Force the AI to ONLY output the target text
    prompt = (
        f"You are a strict spelling data generator. "
        f"Generate a {mode} about {topic} for a 12-year-old (Level {level}). "
        f"IMPORTANT: Output ONLY the {mode} itself. Do not include introductory text, quotes, or explanations."
    )

    try:
        response = model.generate_content(prompt)
        # Clean the text in case Gemini adds markdown or extra spaces
        clean_text = response.text.strip().replace('"', '') 
        return {"text": clean_text}
    except Exception as e:
        print(f"Gemini Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate content")

@app.get("/mistakes")
def get_mistakes():
    return {"mistakes": load_mistakes()}

@app.post("/mistakes")
def track_mistake(payload: MistakeInput):
    mistakes = load_mistakes()
    mistakes.append(payload.word)
    save_mistakes(mistakes)
    return {"message": "Saved", "mistakes": mistakes}