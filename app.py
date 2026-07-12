from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai

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
You are given ONE audio recording.

Your job:

1. Transcribe the audio.
2. If the audio contains or describes a dataset or table,
   reconstruct the dataset.
3. Compute ALL descriptive statistics.

Return ONLY valid JSON.

The JSON MUST contain EXACTLY these keys:

{
  "rows": 0,
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

Rules:

- No markdown.
- No explanation.
- No code fences.
- rows must be an integer.
- columns must be an array.
- mean/std/variance/min/max/median/mode/range/allowed_values/value_range
  must all be JSON objects.
- correlation must be an array.
- If a statistic cannot be computed,
  return an empty object {} or empty array [] as appropriate.
"""

        response = client.models.generate_content(
            # Replace this with a model your API key supports.
            client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"]
            ),
            contents=[
                prompt,
                {
                    "mime_type": "audio/wav",
                    "data": audio_bytes,
                },
            ],
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
