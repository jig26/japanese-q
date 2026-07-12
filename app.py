from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

import base64
import json
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.post("/analyze")
def analyze(req: AudioRequest):

    try:

        audio_bytes = base64.b64decode(req.audio_base64)

        prompt = """
You are given one audio recording.

Listen to it carefully.

If it contains spoken tabular data, reconstruct the table.

Then compute all descriptive statistics.

Return ONLY valid JSON.

Exactly this schema:

{
  "rows": integer,
  "columns": [],
  "mean": {},
  "std": {},
  "variance": {},
  "min": {},
  "max": {},
  "median": {},
  "mode": {},
  "range": {},
  "allowed_values": {},
  "value_range": {},
  "correlation": []
}

Do not explain.
Do not use markdown.
Do not wrap in ```.

Return JSON only.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )
            ]
        )

        text = response.text.strip()

        text = re.sub(r"^```json", "", text)
        text = re.sub(r"^```", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise Exception("Gemini did not return JSON.")

        result = json.loads(match.group())

        required = [
            "rows",
            "columns",
            "mean",
            "std",
            "variance",
            "min",
            "max",
            "median",
            "mode",
            "range",
            "allowed_values",
            "value_range",
            "correlation"
        ]

        output = {}

        for key in required:

            if key in result:
                output[key] = result[key]
            else:
                if key == "rows":
                    output[key] = 0
                elif key in ("columns", "correlation"):
                    output[key] = []
                else:
                    output[key] = {}

        return output

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
