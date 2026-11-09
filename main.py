import os
import time
import uuid
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import yt_dlp

app = FastAPI(title="Multi-Platform Video Summarizer", version="2.0.0")

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

def download_audio(url: str, output_path: str):
    """ดาวน์โหลดเฉพาะเสียงจาก TikTok, FB, IG, หรือ YouTube"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_video(req: SummarizeRequest):
    if not req.video_url:
        raise HTTPException(status_code=400, detail="กรุณาระบุ URL วิดีโอ")

    url = req.video_url.lower()
    
    # 1. กรณีเป็น YouTube: ส่ง URL ตรงให้ Gemini ได้ทันที
    if "youtube.com" in url or "youtu.be" in url:
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=[
                    types.Part.from_uri(file_uri=req.video_url, mime_type="video/*"),
                    "สรุปประเด็นสำคัญของคลิปนี้เป็นภาษาไทยอย่างกระชับ เข้าใจง่าย และแยกเป็นข้อๆ"
                ]
            )
            return SummarizeResponse(summary=response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"YouTube Error: {str(e)}")

    # 2. กรณีเป็น TikTok, Facebook, Instagram: ดึงไฟล์เสียงแล้วส่งให้ Gemini
    temp_filename = f"/tmp/{uuid.uuid4()}.m4a"
    uploaded_file = None

    try:
        # ดาวน์โหลดเสียงจากแพลตฟอร์ม
        download_audio(req.video_url, temp_filename)
        
        # อัปโหลดไฟล์เสียงเข้า Gemini File API
        uploaded_file = client.files.upload(file=temp_filename)
        
        # ส่งให้โมเดลประมวลผล
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=[
                uploaded_file,
                "สรุปประเด็นสำคัญจากเสียงพูดในคลิปนี้เป็นภาษาไทยอย่างกระชับ ชัดเจน และแยกเป็นข้อๆ"
            ]
        )
        return SummarizeResponse(summary=response.text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download/Process Error: {str(e)}")

    finally:
        # ลบไฟล์ในเครื่อง Render และลบไฟล์บน Gemini เพื่อไม่ให้เปลืองพื้นที่
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
