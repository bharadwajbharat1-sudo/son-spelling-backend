# main.py

import json
import os
import random
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# 🔁 In-memory mistakes (simple version)
mistakes = []

@app.get("/generate")
async def generate(
    mode: str = "word",
    level: int = 1,
    topic: str = "general",
    focus_words: Optional[str] = None,
):
    try:
        # 🎯 PRIORITY: reuse weak words
        if focus_words:
            words = [w.strip() for w in focus_words.split(",") if w.strip()]
            if mode == "word":
                return {"text": random.choice(words)}

        if mode == "word":
            prompt = f"""
            Generate ONE spelling word for a 10-12 year old.
            Topic: {topic}
            Difficulty Level: {level}

            Rules:
            - Output ONLY the word
            - No punctuation
            - No sentence
            """

        elif mode == "sentence":
            prompt = f"""
            Generate ONE short sentence for a 10-12 year old.
            Topic: {topic}
            Difficulty Level: {level}

            Rules:
            - Keep it under 8 words
            - Simple and clear
            - Output ONLY the sentence
            """

        else:
            prompt = "Generate a simple word."

        response = model.generate_content(prompt)
        text = response.text.strip().replace('"', '').replace("'", "")

        return {"text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/word-info")
async def word_info(word: str):
    try:
        prompt = f"""
        For the word '{word}', return JSON:
        {{
          "syllables": ["..."],
          "tip": "short memory tip",
          "example": "simple sentence"
        }}
        """

        res = model.generate_content(prompt)
        raw = res.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw)

        return data

    except:
        return {
            "syllables": [word],
            "tip": "",
            "example": ""
        }