from fastapi import FastAPI
from project_ocr.main import app as ocr_app
from project_rag.main import app as rag_app
from project_transcription.main import app as transcription_app
from project_webscrap.main import app as webscrap_app

app = FastAPI(title="Combined AI Services")

app.mount("/ocr", ocr_app)
app.mount("/rag", rag_app)
app.mount("/transcription", transcription_app)
app.mount("/webscrap", webscrap_app)


@app.get("/")
async def root():
    return {"message": "Welcome to Combined AI Services API"}
