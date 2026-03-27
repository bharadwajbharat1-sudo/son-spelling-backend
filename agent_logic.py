import json
import os
import random
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pydantic import BaseModel, Field


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
    if not isinstance(raw, list):
        return []
    return [str(word).strip().lower() for word in raw if str(word).strip()]


def save_mistakes(words: List[str]) -> None:
    normalized = sorted({w.strip().lower() for w in words if w and w.strip()})
    MISTAKES_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def add_mistake(word: str) -> List[str]:
    if not word.strip():
        raise ValueError("Word cannot be empty.")
    mistakes = load_mistakes()
    mistakes.append(word)
    save_mistakes(mistakes)
    return load_mistakes()


def pick_daily_mistakes(count: int = 3) -> List[str]:
    mistakes = load_mistakes()
    if not mistakes:
        return []
    return random.sample(mistakes, k=min(count, len(mistakes)))


def _azure_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not endpoint or not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing Azure OpenAI env vars: AZURE_OPENAI_ENDPOINT and/or AZURE_OPENAI_API_KEY",
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def generate_sentences_with_azure(interests: List[str], mistake_words: List[str]) -> List[str]:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise HTTPException(
            status_code=500,
            detail="Missing Azure OpenAI deployment env var: AZURE_OPENAI_DEPLOYMENT",
        )

    client = _azure_client()
    interests_text = ", ".join(interests)
    mistakes_text = ", ".join(mistake_words) if mistake_words else "None"

    prompt = (
        "You are creating short, fun, child-friendly practice sentences.\n"
        f"Interests: {interests_text}\n"
        f"Words to include exactly once each if provided: {mistakes_text}\n"
        "Return exactly 5 numbered sentences."
    )

    response = client.chat.completions.create(
        model=deployment,
        temperature=0.7,
        messages=[
            {"role": "system", "content": "You write clear, positive educational content for children."},
            {"role": "user", "content": prompt},
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    lines = [line.strip(" -\t") for line in content.splitlines() if line.strip()]
    cleaned = [line.split(".", 1)[1].strip() if line[:2].isdigit() and "." in line else line for line in lines]
    return cleaned[:5]


class MistakeInput(BaseModel):
    word: str = Field(..., description="Misspelled word to track")


class SentenceRequest(BaseModel):
    interests: List[str] = Field(default_factory=lambda: ["FC26", "Science"])


app = FastAPI(title="Kid Practice Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/mistakes")
def get_mistakes() -> dict:
    return {"mistakes": load_mistakes()}


@app.post("/mistakes")
def track_mistake(payload: MistakeInput) -> dict:
    try:
        updated = add_mistake(payload.word)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Mistake saved.", "mistakes": updated}

import time # Add this to your imports at the top

@app.get("/generate")
def generate_dynamic_content(mode: str = "sentence", level: int = 1):
    daily_words = pick_daily_mistakes(2)
    client = _azure_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    timestamp = time.time()

    # Define strict rules based on the Tab (mode) and Level
    if mode == "word":
        complexity = "Provide exactly ONE challenging spelling word. No sentences."
    elif mode == "sentence":
        length = "3-5 words" if level <= 3 else "8-12 words" if level <= 7 else "15+ words"
        complexity = f"Provide exactly ONE sentence of approximately {length}."
    else: # Paragraph mode
        count = "2 short sentences" if level <= 5 else "3-4 complex sentences"
        complexity = f"Provide a paragraph consisting of {count}."

    prompt = f"""
    Task: {complexity}
    Difficulty Level: {level}/10. 
    Topics: FC26 Soccer, Science Facts, or Construction.
    Target words to include: {", ".join(daily_words)}
    Seed: {timestamp}

    CRITICAL RULES:
    1. Return ONLY the raw text. 
    2. At Level {level}, use vocabulary appropriate for that challenge.
    3. Ensure the result is UNIQUE and different from previous generations.
    """

    response = client.chat.completions.create(
        model=deployment,
        temperature=1.0, # Max variety
        messages=[
            {"role": "system", "content": "You are an adaptive spelling tutor. You vary your output significantly based on the level provided."},
            {"role": "user", "content": prompt},
        ],
    )

    generated_text = response.choices[0].message.content.strip().replace('"', '')

    return {
        "text": generated_text,
        "level": level,
        "mode": mode,
        "id": timestamp 
    }