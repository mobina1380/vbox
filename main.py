from fastapi import FastAPI
from modules import speech_emotion_recognition, diarization, audio_segmentation
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/")
def root():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


# All Routes
app.include_router(speech_emotion_recognition.router, prefix="/emotion")
app.include_router(diarization.router, prefix="/diarization")
app.include_router(audio_segmentation.router, prefix="/audio_segmentation")
