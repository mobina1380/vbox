from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio
import torch


model_name = "./whisper-tiny-persian"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


def transcribe_audio(audio_path):

    try:
        audio = Audio(filename=audio_path, sampling_rate=16000)
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_path}' not found.")
        return None

    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    input_features = torch.tensor([input_features])

    generated_ids = model.generate(
        input_features,
        max_length=225,
        num_beams=5,
    )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return transcription


# Example usage (replace with your actual audio file path)
audio_file = "angry.wav"
transcription = transcribe_audio(audio_file)

if transcription:
    print(f"Transcription: {transcription}")
else:
    print("Transcription failed.")
