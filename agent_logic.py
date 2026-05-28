import json
import os
import random
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Kid Practice Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("WARNING: GOOGLE_API_KEY is not set.")
else:
    genai.configure(api_key=google_api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

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


class MistakeInput(BaseModel):
    word: str


@app.get("/health")
async def health_check():
    return {"status": "ok", "provider": "google-gemini"}


@app.get("/generate")
async def generate_dynamic_content(
    mode: str = "sentence",
    level: int = 1,
    topic: str = "anything",
    focus_words: Optional[str] = None,   # NEW: comma-separated words to force into sentence
):
    """
    Generate a spelling practice sentence.
    
    If focus_words is provided (e.g. "because,friend,their"), the AI is
    instructed to naturally include ALL of those exact words in the sentence.
    This ensures struggling words get repeated practice in context.
    """
    if focus_words:
        word_list = [w.strip() for w in focus_words.split(",") if w.strip()]
        # Strong instruction that forces word inclusion
        prompt = (
            f"You are a spelling practice sentence generator for a 12-year-old (Level {level}).\n"
            f"Topic: {topic}\n\n"
            f"CRITICAL REQUIREMENT: You MUST use ALL of these exact words naturally in your sentence: "
            f"{', '.join(word_list)}\n\n"
            f"Rules:\n"
            f"- The sentence must sound natural and make sense\n"
            f"- Use every word from the list exactly as spelled above\n"
            f"- Keep it one sentence, appropriate for a 12-year-old\n"
            f"- Output ONLY the sentence itself — no quotes, no explanation, nothing else"
        )
    else:
        prompt = (
            f"You are a strict spelling data generator. "
            f"Generate a {mode} about {topic} for a 12-year-old (Level {level}). "
            f"IMPORTANT: Output ONLY the {mode} itself. "
            f"Do not include introductory text, quotes, or explanations."
        )

    try:
        response = model.generate_content(prompt)
        clean_text = response.text.strip().replace('"', '').replace("'", "")
        return {"text": clean_text}
    except Exception as e:
        print(f"Gemini Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate content")


@app.get("/word-info")
async def get_word_info(word: str):
    """
    NEW ENDPOINT: Returns syllable breakdown, pronunciation tip, and
    an example sentence for the Word Builder drill mode.
    """
    prompt = (
        f"For the English word '{word}', provide:\n"
        f"1. syllables: split the word into syllables as a JSON array of strings\n"
        f"2. tip: one short memory trick or spelling tip (max 12 words)\n"
        f"3. example: one simple sentence using the word (suitable for a 12-year-old)\n\n"
        f"Respond ONLY with valid JSON in this exact format, nothing else:\n"
        f'{{"syllables": ["syl","la","ble"], "tip": "...", "example": "..."}}'
    )
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown code fences if Gemini adds them
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {
            "word": word,
            "syllables": data.get("syllables", [word]),
            "tip": data.get("tip", ""),
            "example": data.get("example", ""),
        }
    except Exception as e:
        print(f"Word info error for '{word}': {e}")
        # Fallback: basic syllabification
        return {
            "word": word,
            "syllables": [word],
            "tip": "",
            "example": "",
        }


@app.get("/mistakes")
def get_mistakes():
    return {"mistakes": load_mistakes()}


@app.post("/mistakes")
def track_mistake(payload: MistakeInput):
    mistakes = load_mistakes()
    mistakes.append(payload.word)
    save_mistakes(mistakes)
    return {"message": "Saved", "mistakes": mistakes}
