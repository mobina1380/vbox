#question
from fastapi import FastAPI, UploadFile, File, HTTPException
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
import tempfile
from fastapi import APIRouter

router = APIRouter()


app = FastAPI()

use_auth_token = "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"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=use_auth_token)

def diarization(audio_path: str):
    try:
        diarization_result = pipeline(audio_path)
        return diarization_result

    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during diarization: " + str(e))
    

@router.post("/")
def process_diarization(short_audio: UploadFile = File(...), long_audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_short:
        temp_short.write(short_audio.file.read())
        short_audio_path = temp_short.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_long:
        temp_long.write(long_audio.file.read())
        long_audio_path = temp_long.name

    try:
        short_audio_segment = AudioSegment.from_file(short_audio_path)
        long_audio_segment = AudioSegment.from_file(long_audio_path)
        combined_audio = short_audio_segment + long_audio_segment
        combined_audio_path = "combined_audio.wav"
        combined_audio.export(combined_audio_path, format="wav")
        diarization_result = diarization(combined_audio_path)
        first_speaker_id = None
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if first_speaker_id is None:
                first_speaker_id = speaker
                break
        result = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if speaker == first_speaker_id:
                result.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        Allresult = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            Allresult.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred: " + str(e))

    finally:
        os.unlink(short_audio_path)
        os.unlink(long_audio_path)
        os.unlink(combined_audio_path)  

    return {"Short speaker times in long voice": result , "Allresult":Allresult}
