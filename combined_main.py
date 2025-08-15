from fastapi import FastAPI
from Project_OCR.main import app as ocr_app
from Project_RAG.main import app as rag_app
from Project_Transcription.main import app as transcription_app
from Project_Webscrap.main import app as webscrap_app

app = FastAPI(title="Combined AI Services")

# Mount each project under its own route
app.mount("/OCR", ocr_app)
app.mount("/RAG", rag_app)
app.mount("/Transcription", transcription_app)
app.mount("/Webscrap", webscrap_app)

@app.get("/")
async def root():
    return {"message": "Welcome to Combined AI Services API"}
