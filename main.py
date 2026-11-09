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

app = FastAPI(title="Multi-Platform Video Summarizer API", version="2.0.0")

# เปิดสิทธิ์ CORS ให้ LangMind สามารถส่งคำขอเข้ามาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ดึง Gemini API Key จาก Environment Variable บน Render
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class SummarizeRequest(BaseModel):
    video_url: str

class SummarizeResponse(BaseModel):
    summary: str

def download_audio_safe(url: str, output_path: str):
    """
    ดาวน์โหลดเฉพาะเสียงจาก TikTok, Facebook, Instagram 
    พร้อมตั้งค่า Headers เลียนแบบเบราว์เซอร์มือถือเพื่อป้องกันการบล็อก
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tiktok.com/',
        },
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def summarize_with_gemini(contents_payload: list) -> str:
    """เรียกประมวลผล Gemini พร้อมระบบสลับโมเดลอัตโนมัติหากติด 503"""
    models = ['gemini-3.5-flash-lite', 'gemini-3.8-flash']
    last_err = None

    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents_payload
                )
                return response.text
            except Exception as e:
                last_err = e
                err_text = str(e)
                print(f"[{model_name}] attempt {attempt + 1} failed: {err_text}")
                if "503" in err_text or "UNAVAILABLE" in err_text:
                    time.sleep(2)
                    continue
                break

    raise last_err

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Multi-Platform Summarizer API is running"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_video(req: SummarizeRequest):
    if not req.video_url:
        raise HTTPException(status_code=400, detail="กรุณาระบุ URL วิดีโอ")

    url = req.video_url.strip()
    url_lower = url.lower()
    prompt_text = "สรุปประเด็นสำคัญของเนื้อหานี้เป็นภาษาไทยอย่างกระชับ ชัดเจน แยกเป็นหัวข้อและข้อย่อยให้อ่านเข้าใจง่าย"

    # กรณีที่ 1: เป็นคลิป YouTube (ประมวลผลผ่าน URI ได้โดยตรง ไม่ต้องโหลดไฟล์)
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        try:
            summary = summarize_with_gemini([
                types.Part.from_uri(file_uri=url, mime_type="video/*"),
                prompt_text
            ])
            return SummarizeResponse(summary=summary)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"YouTube Error: {str(e)}")

    # กรณีที่ 2: TikTok, Facebook Reels, Instagram Reels
    temp_file = f"/tmp/{uuid.uuid4()}.m4a"
    gemini_file = None

    try:
        # 1. ดึงเฉพาะไฟล์เสียง (ไม่เกิน 1-3 MB)
        download_audio_safe(url, temp_file)

        # 2. ส่งไฟล์เสียงขึ้นไปให้ Gemini ประมวลผล
        gemini_file = client.files.upload(file=temp_file)

        # 3. สั่งให้สรุปจากเสียงพูด
        summary = summarize_with_gemini([
            gemini_file,
            prompt_text
        ])
        return SummarizeResponse(summary=summary)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download/Process Error: {str(e)}")

    finally:
        # ล้างไฟล์ชั่วคราวทิ้งทุกครั้งเพื่อประหยัด RAM/Disk
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

        if gemini_file:
            try:
                client.files.delete(name=gemini_file.name)
            except Exception:
                pass
