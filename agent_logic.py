import json
import os
import random
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="Kid Practice Backend", version="1.0.0")

# UPDATED: Allow both local testing and your Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MISTAKES_FILE = BASE_DIR / "mistakes.json"

def _ensure_mistakes_file() -> None:
    if not MISTAKES_FILE.exists():
        MISTAKES_FILE.write_text("[]", encoding="utf-8")

def load_mistakes() -> List[str]:
    _ensure_mistakes_file()
    try:
        raw = json.loads(MISTAKES_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raw = []
    return [str(word).strip().lower() for word in raw if str(word).strip()]

def save_mistakes(words: List[str]) -> None:
    normalized = sorted({w.strip().lower() for w in words if w and w.strip()})
    MISTAKES_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")

def add_mistake(word: str) -> List[str]:
    mistakes = load_mistakes()
    mistakes.append(word)
    save_mistakes(mistakes)
    return mistakes

def pick_daily_mistakes(count: int = 3) -> List[str]:
    mistakes = load_mistakes()
    if not mistakes:
        return []
    return random.sample(mistakes, k=min(count, len(mistakes)))

def _openai_client() -> OpenAI:
    # This looks for the key we will put in Render.com
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

class MistakeInput(BaseModel):
    word: str

@app.get("/generate")
def generate_dynamic_content(mode: str = "sentence", level: int = 1, topic: str = "anything"):
    client = _openai_client()
    
    # Use the 'topic' from the UI in the prompt
    prompt = f"Create a {mode} at difficulty level {level}/10. The topic must be about: {topic}. " \
             f"Make it exciting for a 12-year-old."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fun, encouraging spelling tutor."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {"text": response.choices[0].message.content.strip()}

@app.get("/mistakes")
def get_mistakes():
    return {"mistakes": load_mistakes()}

@app.post("/mistakes")
def track_mistake(payload: MistakeInput):
    updated = add_mistake(payload.word)
    return {"message": "Saved", "mistakes": updated}