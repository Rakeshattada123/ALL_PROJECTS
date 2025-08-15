# main.py

import os
import uuid
import datetime
import re
import tempfile
from contextlib import asynccontextmanager

import torch
import yt_dlp
import whisper
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

# --- Configuration & Global Variables ---

# Load environment variables from .env file
load_dotenv()

# Dictionary to store ML models, loaded at startup
MODELS = {}
# Dictionary to store the status and results of tasks
TASKS = {}

# --- Pydantic Models for API Validation ---

class ProcessRequest(BaseModel):
    youtube_url: str = Field(..., description="The URL of the YouTube video to process.", example="https://www.youtube.com/watch?v=lXIM6krQo8M")
    target_languages: list[str] = Field(..., description="A list of languages to translate the transcript into.", example=["Hindi","English","Telugu"])
    whisper_model_size: str = Field("medium", description="The size of the Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large').")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ResultData(BaseModel):
    metadata: dict
    original_transcript: str
    translations: dict[str, str]

class StatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result: ResultData | None = None

# --- Helper Functions (from the notebook) ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds into a user-friendly HH:MM:SS format."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_emotion(text: str) -> str:
    """
    Analyzes text to determine an emotion emoji.
    First checks for custom keywords, then uses the AI model.
    """
    keyword_emotion_map = {
        "🙏": ["thank you", "thanks", "grateful", "appreciate it"],
        "🤔": ["i wonder", "let me think", "what if"],
        "😂": ["laughing", "so funny", "lol"],
        "✅": ["that's correct", "exactly", "i agree", "deal"],
    }
    emotion_map = {
        "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😨",
        "surprise": "😮", "disgust": "🤢", "neutral": "😐"
    }
    
    # 1. Check for custom keywords first
    lower_text = text.lower()
    for emoji, keywords in keyword_emotion_map.items():
        if any(keyword.lower() in lower_text for keyword in keywords):
            return emoji

    # 2. If no keyword match, use the AI model
    if MODELS.get("emotion_classifier"):
        prediction = MODELS["emotion_classifier"](text)[0]
        emotion_label = prediction['label']
        return emotion_map.get(emotion_label, "❓")
    
    return "😐" # Default if model not loaded

# --- Core Processing Logic ---

def process_video_pipeline(task_id: str, req: ProcessRequest):
    """The main background task that runs the full download-transcribe-translate pipeline."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            TASKS[task_id] = {"status": "processing", "message": "Downloading audio...", "result": None}
            
            # 1. Download Audio and Extract Metadata
            audio_filename = os.path.join(temp_dir, "downloaded_audio.mp3")
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
                'outtmpl': os.path.join(temp_dir, 'downloaded_audio.%(ext)s'),
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(req.youtube_url, download=True)
                video_metadata = {
                    'title': info_dict.get('title', 'N/A'),
                    'uploader': info_dict.get('uploader', 'N/A'),
                    'duration': str(datetime.timedelta(seconds=info_dict.get('duration', 0))),
                    'published_date': datetime.datetime.strptime(info_dict.get('upload_date', ''), '%Y%m%d').strftime('%Y-%m-%d') if info_dict.get('upload_date') else 'N/A',
                    'hashtags': ', '.join([f"#{h}" for h in info_dict.get('hashtags', [])]),
                    'description': info_dict.get('description', '')
                }

            # 2. Transcribe with Whisper and add Emotion
            TASKS[task_id]["message"] = "Transcribing and analyzing emotions..."
            whisper_model = MODELS.get("whisper_model")
            if not whisper_model:
                raise RuntimeError("Whisper model not loaded.")
            
            transcribe_result = whisper_model.transcribe(audio_filename, verbose=False)

            # Build rich metadata header
            header = (
                f"--- VIDEO METADATA ---\n"
                f"Title: {video_metadata['title']}\n"
                f"Uploader: {video_metadata['uploader']}\n"
                f"Published Date: {video_metadata['published_date']}\n"
                f"Duration: {video_metadata['duration']}\n"
                f"Hashtags: {video_metadata['hashtags']}\n\n"
                f"Description:\n{video_metadata['description']}\n"
                f"--- END METADATA ---\n\n"
                f"--- TRANSCRIPT WITH EMOTION ANALYSIS ---\n\n"
            )
            
            transcript_lines = []
            for segment in transcribe_result["segments"]:
                text = segment["text"].strip()
                if not text: continue
                
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                emoji = get_emotion(text)
                line = f"[{start_time} --> {end_time}] {emoji} {text}"
                transcript_lines.append(line)
            
            full_transcript = header + "\n".join(transcript_lines)

            # 3. Translate with Gemini
            TASKS[task_id]["message"] = "Translating transcript..."
            translation_model = genai.GenerativeModel('gemini-1.5-pro-latest')
            translations = {}
            
            for lang in req.target_languages:
                TASKS[task_id]["message"] = f"Translating to {lang}..."
                prompt = (
                    f"You are an expert translator. The following text is in English. "
                    f"Translate it accurately into {lang}. IMPORTANT: Preserve the timestamps "
                    f"like [HH:MM:SS --> HH:MM:SS], all emojis (like 🙏, 😄, 😐), and the "
                    f"overall structure, including the header information.\n\n"
                    f"--- TEXT TO TRANSLATE ---\n\n{full_transcript}"
                )
                response = translation_model.generate_content(prompt)
                translations[lang] = response.text
            
            # 4. Finalize Task
            final_result = ResultData(
                metadata=video_metadata,
                original_transcript=full_transcript,
                translations=translations
            )
            TASKS[task_id] = {
                "status": "completed",
                "message": "Processing complete.",
                "result": final_result
            }

    except Exception as e:
        TASKS[task_id] = {"status": "failed", "message": f"An error occurred: {str(e)}", "result": None}

# --- FastAPI Application Setup & Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and clear on shutdown."""
    print("--- Server starting up. Loading models... ---")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Whisper Model
    # Using 'base' for faster loading on CPU. Change to 'medium' or 'large' for better accuracy.
    # whisper_model_size = "base"
    MODELS["whisper_model"] = whisper.load_model("base", device=device)
    print("✅ Whisper model loaded.")
    
    # Load Emotion Classifier Model
    MODELS["emotion_classifier"] = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if device == "cuda" else -1)
    print("✅ Emotion classifier model loaded.")
    
    # Configure Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured.")
    
    print("--- Models loaded. Server is ready. ---")
    yield
    
    # Clean up models on shutdown
    print("--- Server shutting down. Clearing models. ---")
    MODELS.clear()

app = FastAPI(lifespan=lifespan, title="Video Transcription and Translation API")

# --- API Endpoints ---

@app.get("/", summary="Root endpoint for health check")
def read_root():
    return {"status": "ok", "message": "Welcome to the Video Processing API"}

@app.post("/process-video", response_model=TaskResponse, status_code=202, summary="Start video processing")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Accepts a YouTube URL and target languages, then starts the processing pipeline in the background.
    """
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "pending", "message": "Task received. Queued for processing.", "result": None}
    
    # Start the long-running task in the background
    background_tasks.add_task(process_video_pipeline, task_id, req)
    
    return {"task_id": task_id, "status": "pending", "message": "Processing started in the background."}

@app.get("/status/{task_id}", response_model=StatusResponse, summary="Check the status of a task")
async def get_task_status(task_id: str):
    """
    Poll this endpoint with the task_id to get the current status and the final result.
    """
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return StatusResponse(task_id=task_id, **task)