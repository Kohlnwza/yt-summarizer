import os
import json
import time
import uuid
import traceback
import urllib.request
import urllib.parse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import yt_dlp

app = FastAPI(title="Multi-Platform Video Summarizer API", version="2.1.0")

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

def download_tiktok_audio_direct(url: str, output_path: str):
    """ดึงเสียงจาก TikTok ผ่าน TikWM เพื่อข้ามระบบบล็อก Data Center IP"""
    api_url = f"https://www.tikwm.com/api/?url={urllib.parse.quote(url)}"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        
    if data.get("code") == 0 and "data" in data:
        # เลือกเอาไฟล์เสียงเพลง/เสียงพูด ถ้าไม่มีให้ดึงเสียงจากวิดีโอ
        audio_url = data["data"].get("music") or data["data"].get("play")
        audio_req = urllib.request.Request(audio_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(audio_req, timeout=20) as a_resp, open(output_path, 'wb') as f:
            f.write(a_resp.read())
        return True
    else:
        raise Exception(f"TikWM Error: {data.get('msg', 'Failed to fetch TikTok data')}")

def download_generic_audio(url: str, output_path: str):
    """สำหรับ Facebook, Instagram หรือแพลตฟอร์มอื่นๆ ผ่าน yt-dlp"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'nocheckcertificate': True,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1',
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def summarize_with_gemini(contents_payload: list) -> str:
    """ประมวลผลด้วย Gemini 3.5 Flash Lite และมีระบบ Retry"""
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

@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_video(req: SummarizeRequest):
    if not req.video_url:
        raise HTTPException(status_code=400, detail="กรุณาระบุ URL วิดีโอ")

    url = req.video_url.strip()
    url_lower = url.lower()
    prompt_text = "สรุปประเด็นสำคัญของเนื้อหานี้เป็นภาษาไทยอย่างกระชับ ชัดเจน แยกเป็นหัวข้อและข้อย่อยให้อ่านเข้าใจง่าย"

    # 1. YouTube: ยิงตรงหา Gemini ทันที ไม่ต้องโหลดไฟล์
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

    # 2. TikTok / FB / IG: โหลดไฟล์เสียงแล้วส่งให้ Gemini
    temp_file = f"/tmp/{uuid.uuid4()}.mp3"
    gemini_file = None

    try:
        if "tiktok.com" in url_lower:
            download_tiktok_audio_direct(url, temp_file)
        else:
            download_generic_audio(url, temp_file)

        # อัปโหลดไฟล์เสียงเข้า Gemini File API
        gemini_file = client.files.upload(file=temp_file)

        # สรุปเนื้อหาจากเสียง
        summary = summarize_with_gemini([
            gemini_file,
            prompt_text
        ])
        return SummarizeResponse(summary=summary)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download/Process Error: {str(e)}")

    finally:
        # เคลียร์ไฟล์ชั่วคราว
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
