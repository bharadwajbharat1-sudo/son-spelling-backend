import json
import os
import random
import re
from typing import Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    import google.generativeai as genai
except ImportError:  # App still works without Gemini.
    genai = None

app = FastAPI(title="Active Spelling Retrieval API", version="2.0.0")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if allowed_origins == ["*"] else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = None

if genai is not None and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)

# Curated words are the reliable default. They specifically reinforce common
# spelling challenges: short vowels, silent-e, endings, function words,
# consonant blends, vowel teams, multisyllable words, and commonly confused words.
WORD_BANK: Dict[int, List[Dict[str, object]]] = {
    1: [
        {"word": "the", "syllables": ["the"], "tip": "Say: th + uh. It is a small but important word.", "example": "The dog ran home.", "pattern": "function word"},
        {"word": "was", "syllables": ["was"], "tip": "Remember: w-a-s, even though it may sound like wuz.", "example": "He was very kind.", "pattern": "function word"},
        {"word": "his", "syllables": ["his"], "tip": "End with s: h-i-s.", "example": "His bag is blue.", "pattern": "final sound"},
        {"word": "from", "syllables": ["from"], "tip": "Hear the blend fr at the start.", "example": "She came from school.", "pattern": "consonant blend"},
        {"word": "with", "syllables": ["with"], "tip": "Start with w and finish with th.", "example": "I went with Dad.", "pattern": "function word"},
        {"word": "went", "syllables": ["went"], "tip": "Tap every sound: w-e-n-t.", "example": "We went to class.", "pattern": "sound sequencing"},
        {"word": "stop", "syllables": ["stop"], "tip": "Hear both sounds in st.", "example": "Stop at the line.", "pattern": "consonant blend"},
        {"word": "best", "syllables": ["best"], "tip": "Do not drop the final t.", "example": "Try your best today.", "pattern": "final consonant"},
    ],
    2: [
        {"word": "made", "syllables": ["made"], "tip": "Silent e makes a say its name.", "example": "She made a card.", "pattern": "silent e"},
        {"word": "smile", "syllables": ["smile"], "tip": "Silent e makes i long: smile.", "example": "His smile was bright.", "pattern": "silent e"},
        {"word": "became", "syllables": ["be", "came"], "tip": "Split it: be + came.", "example": "The sky became dark.", "pattern": "syllable chunking"},
        {"word": "letter", "syllables": ["let", "ter"], "tip": "Double t in the middle.", "example": "I wrote a letter.", "pattern": "double consonant"},
        {"word": "smelled", "syllables": ["smelled"], "tip": "Start with smell, then add ed.", "example": "The flower smelled sweet.", "pattern": "word ending"},
        {"word": "fluttered", "syllables": ["flut", "tered"], "tip": "Build it: flutter + ed.", "example": "The flag fluttered softly.", "pattern": "word ending"},
        {"word": "noise", "syllables": ["noise"], "tip": "The vowel team oi says oy.", "example": "I heard a loud noise.", "pattern": "vowel team"},
        {"word": "their", "syllables": ["their"], "tip": "Their has heir inside it.", "example": "Their house is nearby.", "pattern": "confused word"},
    ],
    3: [
        {"word": "advantage", "syllables": ["ad", "van", "tage"], "tip": "Say and type each chunk: ad-van-tage.", "example": "Practice gives you an advantage.", "pattern": "multisyllable"},
        {"word": "exceptional", "syllables": ["ex", "cep", "tion", "al"], "tip": "Build it in four chunks.", "example": "She did an exceptional job.", "pattern": "multisyllable"},
        {"word": "different", "syllables": ["dif", "fer", "ent"], "tip": "Double f, then end with ent.", "example": "Each person is different.", "pattern": "double consonant"},
        {"word": "important", "syllables": ["im", "por", "tant"], "tip": "Listen for three beats: im-por-tant.", "example": "Reading is important.", "pattern": "multisyllable"},
        {"word": "beautiful", "syllables": ["beau", "ti", "ful"], "tip": "Remember: beau + ti + ful.", "example": "The garden looked beautiful.", "pattern": "vowel pattern"},
        {"word": "remember", "syllables": ["re", "mem", "ber"], "tip": "Say every chunk slowly.", "example": "Remember to check your work.", "pattern": "multisyllable"},
        {"word": "because", "syllables": ["be", "cause"], "tip": "Big Elephants Can Always Understand Small Elephants.", "example": "I smiled because I won.", "pattern": "memory word"},
        {"word": "sentence", "syllables": ["sen", "tence"], "tip": "It starts with sent, but ends with ence.", "example": "Write one clear sentence.", "pattern": "word ending"},
    ],
}


def clean_word(value: str) -> str:
    return re.sub(r"[^A-Za-z'-]", "", value).strip().lower()


def lookup_word(word: str) -> Optional[Dict[str, object]]:
    target = clean_word(word)
    for entries in WORD_BANK.values():
        for item in entries:
            if item["word"] == target:
                return item
    return None


def fallback_info(word: str) -> Dict[str, object]:
    clean = clean_word(word) or word.strip()
    return {
        "word": clean,
        "syllables": [clean],
        "tip": "Look, say, cover, type, and check the word.",
        "example": f"I can spell {clean}.",
        "pattern": "practice word",
    }


def parse_json_response(raw: str) -> Dict[str, object]:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


@app.get("/")
def root():
    return {"message": "Active Spelling Retrieval API is running", "ai_enabled": bool(model)}


@app.get("/health")
def health():
    return {"status": "ok", "ai_enabled": bool(model), "model": MODEL_NAME if model else None}


@app.get("/generate")
def generate(
    mode: str = Query("word", pattern="^(word|sentence)$"),
    level: int = Query(1, ge=1, le=3),
    topic: str = "general",
    focus_words: Optional[str] = None,
):
    if focus_words:
        candidates = [clean_word(word) for word in focus_words.split(",")]
        candidates = [word for word in candidates if word]
        if candidates:
            chosen = random.choice(candidates)
            if mode == "word":
                return {"text": chosen, "source": "review"}
            return {"text": f"Please use {chosen} in a sentence.", "source": "review"}

    if mode == "word":
        item = random.choice(WORD_BANK[level])
        return {
            "text": item["word"],
            "pattern": item["pattern"],
            "source": "curated",
        }

    sentence_bank = {
        1: ["The dog ran home.", "He was very kind.", "We went to class."],
        2: ["She made a bright card.", "The flag fluttered softly.", "Their house is nearby."],
        3: ["Practice gives you an advantage.", "Reading is very important.", "Write one clear sentence."],
    }
    return {"text": random.choice(sentence_bank[level]), "source": "curated"}


@app.get("/word-info")
def word_info(word: str):
    clean = clean_word(word)
    if not clean:
        return fallback_info("word")

    known = lookup_word(clean)
    if known:
        return {"word": clean, **known}

    if model is not None:
        try:
            prompt = f"""
Return valid JSON only for the spelling word {json.dumps(clean)}.
Use this exact schema:
{{
  "word": {json.dumps(clean)},
  "syllables": ["short chunks"],
  "tip": "one short, concrete spelling tip under 14 words",
  "example": "one simple sentence under 9 words",
  "pattern": "short spelling pattern label"
}}
The learner is 12 and benefits from simple, age-respectful instructions.
"""
            response = model.generate_content(prompt)
            data = parse_json_response(response.text)
            if isinstance(data.get("syllables"), list):
                return data
        except Exception:
            pass

    return fallback_info(clean)
