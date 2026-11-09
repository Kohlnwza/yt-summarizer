import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="YouTube Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class SummarizeRequest(BaseModel):
    video_url: str

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_youtube_video(req: SummarizeRequest):
    if not req.video_url:
        raise HTTPException(status_code=400, detail="กรุณาระบุ URL ของวิดีโอ YouTube")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=[
                types.Part.from_uri(
                    file_uri=req.video_url,
                    mime_type="video/*",
                ),
                "สรุปเนื้อหาสำคัญของวิดีโอนี้เป็นภาษาไทยอย่างกระชับ เข้าใจง่าย และแยกเป็นข้อๆ"
            ]
        )
        return SummarizeResponse(summary=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
