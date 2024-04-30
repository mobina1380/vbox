# question 2
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Header
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import io
import zipfile
from fastapi import APIRouter


app = FastAPI()
router = APIRouter()


def split_audio(
    audio_segment, min_silence_len=1000, silence_thresh=-30, segment_duration=5000
):
    try:
        segments = split_on_silence(
            audio_segment,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
        )
        final_segments = []
        current_segment = AudioSegment.empty()
        silence_length = 0

        for seg in segments:
            if silence_length + len(current_segment) < segment_duration:
                current_segment += seg
            else:
                if len(current_segment) >= segment_duration:
                    final_segments.append(current_segment)
                current_segment = seg
                silence_length = 0
            silence_length += len(seg)
            
        if len(current_segment) >= segment_duration:
            final_segments.append(current_segment)

        return final_segments

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting audio: {str(e)}")


@router.post("/")
async def split_audio_endpoint(
    file: UploadFile = File(...), auth_token: str = Header(...)
):
    if auth_token != "vbox":
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    try:
        contents = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(contents))
        audio_array = np.array(audio.get_array_of_samples())
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels,
        )
        segments = split_audio(audio_segment)
        all_segments_files = []
        for i, segment in enumerate(segments):
            segment_file = io.BytesIO()
            segment.export(segment_file, format="wav")
            segment_file.seek(0)  
            all_segments_files.append(
                {
                    "segment_name": f"segment_{i+1}.wav",
                    "segment_data": segment_file.getvalue(),
                }
            )

        # Create a zip file containing all segments
        zip_file = io.BytesIO()
        with io.BytesIO() as temp_zip:
            with zipfile.ZipFile(temp_zip, mode="w") as zipf:
                for segment in all_segments_files:
                    zipf.writestr(segment["segment_name"], segment["segment_data"])
            zip_file.write(temp_zip.getvalue())

        return Response(
            content=zip_file.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=segments.zip"},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")





# import webrtcvad
# import wave
# import numpy as np

# # Function to read audio file
# def read_wave(path):
#     with wave.open(path, 'rb') as wf:
#         sample_width = wf.getsampwidth()
#         sample_rate = wf.getframerate()
#         n_channels = wf.getnchannels()
#         pcm_data = wf.readframes(wf.getnframes())
#     return pcm_data, sample_rate, sample_width, n_channels

# # Function to split audio based on VAD
# def split_audio_vad(audio, sample_rate, vad_level=0.3, silence_duration=5000):
#     vad = webrtcvad.Vad()
#     vad.set_mode(3)  # 0: Aggressive, 1: Low, 2: Medium, 3: High
#     frame_duration = 30  # ms

#     # Convert audio to 16-bit signed PCM
#     audio_data = np.frombuffer(audio, dtype=np.int16)

#     # Split audio based on VAD
#     vad_segments = []
#     segment_start = 0
#     segment_end = 0
#     in_speech = False
#     frame_size = int(sample_rate * (frame_duration / 1000.0))
#     for i in range(0, len(audio_data), frame_size):
#         frame = audio_data[i:i + frame_size]
#         if len(frame) < frame_size:
#             frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
#         if vad.is_speech(frame.tobytes(), sample_rate):
#             if not in_speech:
#                 segment_start = i / sample_rate
#                 if segment_end - segment_start >= silence_duration / 1000.0:
#                     vad_segments.append((segment_start, segment_end))
#                 in_speech = True
#             segment_end = i / sample_rate
#         else:
#             if in_speech:
#                 segment_end = i / sample_rate
#                 vad_segments.append((segment_start, segment_end))
#                 in_speech = False

#     # If speech continues till the end of the audio
#     if in_speech:
#         vad_segments.append((segment_start, segment_end))

#     return vad_segments

# # Function to write audio segment to a new WAV file
# def write_segment_to_wav(segment, sample_rate, audio_data, output_file):
#     start_sample = int(segment[0] * sample_rate)
#     end_sample = int(segment[1] * sample_rate)
#     segment_data = audio_data[start_sample:end_sample]

#     with wave.open(output_file, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)  # 16-bit
#         wf.setframerate(sample_rate)
#         wf.writeframes(segment_data)

# # Example usage
# audio_path = "longvoice.wav"
# pcm_data, sample_rate, sample_width, n_channels = read_wave(audio_path)
# vad_segments = split_audio_vad(pcm_data, 16000)

# # Output segments to separate files
# for i, segment in enumerate(vad_segments):
#     output_file = f"segment_{i+1}.wav"
#     write_segment_to_wav(segment, sample_rate, pcm_data, output_file)
#     print(f"Segment {i+1} written to {output_file}")
