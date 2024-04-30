# question 5
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Header
from pydub import AudioSegment
import numpy as np
from fastapi import APIRouter
import tempfile


import io
from modules.preprocess import predict

app = FastAPI()

model_name_or_path = "m3hrdadfi/hubert-base-persian-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
router = APIRouter()

@router.post("/")
async def emotion_recognition(
    file: UploadFile = File(...), auth_token: str = Header(...)
):
    # Check if the provided auth token is correct

    # Read uploaded audio file
    contents = await file.read()
    audio = AudioSegment.from_file(io.BytesIO(contents))

    # Convert AudioSegment to numpy array for librosa processing
    audio_array = np.array(audio.get_array_of_samples())
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_file.write(contents)
        temp_audio_file_path = temp_audio_file.name
    outputs = predict(temp_audio_file_path, sampling_rate) 
    return outputs