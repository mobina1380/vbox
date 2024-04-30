import torchaudio
from transformers import HubertForSequenceClassification
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/hubert-base-persian-speech-emotion-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model = HubertForSequenceClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(path):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path)
    inputs = feature_extractor(
        speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    for i, score in enumerate(scores):
        if config.id2label[i] == "Anger":
            outputs = [
                {"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"}
            ]
            return outputs
