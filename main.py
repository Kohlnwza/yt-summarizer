import os
import traceback
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

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_youtube_video(req: SummarizeRequest):
    if not req.video_url:
        raise HTTPException(status_code=400, detail="กรุณาระบุ URL ของวิดีโอ YouTube")
    
    # กำหนดโมเดลสำรองหากโมเดลหลักติดคิว 503
    models_to_try = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-flash']
    last_error = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_uri(
                        file_uri=req.video_url,
                        mime_type="video/*",
                    ),
                    "สรุปเนื้อหาสำคัญของคลิป YouTube นี้เป็นภาษาไทยอย่างกระชับ สรุปเป็นประเด็นหลักและข้อย่อยให้เข้าใจง่าย"
                ]
            )
            return SummarizeResponse(summary=response.text)
        except Exception as e:
            last_error = e
            print(f"Failed with {model_name}: {e}")
            continue

    traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(last_error))
