from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

import base64
import io
import json
import os
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


@app.post("/analyze")
def analyze(req: AudioRequest):

    try:

        # --------------------------------------------------
        # Decode base64 audio
        # --------------------------------------------------

        audio_bytes = base64.b64decode(req.audio_base64)

        audio = io.BytesIO(audio_bytes)

        # --------------------------------------------------
        # Ask Gemini to transcribe and extract a table
        # --------------------------------------------------

        prompt = """
You will receive an audio recording.

Transcribe it.

If the speaker is describing a table or dataset,
return ONLY JSON in this format:

{
  "rows":[
    {
      "column1": value,
      "column2": value
    }
  ]
}

Return ONLY JSON.
"""

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                prompt,
                {
                    "mime_type": "audio/wav",
                    "data": audio_bytes,
                },
            ],
        )

        text = response.text.strip()

        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)

        df = pd.DataFrame(data["rows"])

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        numeric = df.select_dtypes(include=np.number)

        result = {
            "rows": len(df),
            "columns": list(df.columns),
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

        # Numeric statistics

        for col in numeric.columns:

            result["mean"][col] = float(numeric[col].mean())

            result["std"][col] = float(numeric[col].std())

            result["variance"][col] = float(numeric[col].var())

            result["min"][col] = float(numeric[col].min())

            result["max"][col] = float(numeric[col].max())

            result["median"][col] = float(numeric[col].median())

            result["range"][col] = (
                float(numeric[col].max())
                - float(numeric[col].min())
            )

            result["value_range"][col] = [
                float(numeric[col].min()),
                float(numeric[col].max())
            ]

            m = numeric[col].mode()

            if len(m):

                result["mode"][col] = float(m.iloc[0])

            else:

                result["mode"][col] = None

        # Categorical columns

        for col in df.columns:

            if col not in numeric.columns:

                vals = list(df[col].dropna().unique())

                result["allowed_values"][col] = vals

        # Correlation matrix

        if len(numeric.columns) >= 2:

            corr = numeric.corr()

            for c1 in corr.columns:

                for c2 in corr.columns:

                    if c1 != c2:

                        result["correlation"].append({

                            "column1": c1,

                            "column2": c2,

                            "value": float(corr.loc[c1, c2])

                        })

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )