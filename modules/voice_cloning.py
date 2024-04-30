#This code is not complete
import torch
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import os
from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.io import load_checkpoint
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.tts.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.vocoder.utils.audio import save_wav, inv_preemphasis
from TTS.tts.utils.text.cleaners import custom_arpabet

def load_model():
    # load configs
    TTS_MODEL = "tacotron2"  # 'tacotron2' or 'tacotron2-DDC'
    TTS_CONFIG = './TTS/tts/configs/config.json'
    TTS_CHECKPOINT = './pretrained_models/tacotron2_130k_farsi.pth.tar'
    TTS_CONFIG = load_config(TTS_CONFIG)

    ap = AudioProcessor(**TTS_CONFIG.audio)
    # LOAD TTS MODEL
    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(phonemes), ap, TTS_CONFIG)
    if TTS_CONFIG.use_pretrained:
        if torch.cuda.is_available():
            cp = torch.load(TTS_CHECKPOINT)
        else:
            cp = torch.load(TTS_CHECKPOINT, map_location=torch.device('cpu'))
        model.load_state_dict(cp['model'])
    else:
        model.load_state_dict(torch.load(TTS_CHECKPOINT, map_location=torch.device('cpu')))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model, ap

def load_vocoder():
    # LOAD VOCODER
    VOCODER_MODEL = "wavernn"
    VOCODER_CONFIG = "./TTS/vocoder/configs/config.json"
    VOCODER_CHECKPOINT = "./pretrained_models/wavernn_farsi_10k.pth.tar"
    # load config
    vocoder_config = load_config(VOCODER_CONFIG)
    # load the audio processor
    vocoder_ap = AudioProcessor(**vocoder_config['audio'])
    # LOAD VOCODER MODEL
    vocoder = setup_generator(VOCODER_MODEL, vocoder_ap)
    if torch.cuda.is_available():
        vocoder.load_state_dict(torch.load(VOCODER_CHECKPOINT))
        vocoder.cuda()
    else:
        vocoder.load_state_dict(torch.load(VOCODER_CHECKPOINT, map_location=torch.device('cpu')))
    vocoder.eval()
    return vocoder, vocoder_ap

def tts(model, ap, text):
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, TTS_CONFIG, use_cuda=torch.cuda.is_available())
    if torch.cuda.is_available():
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    waveform = inv_preemphasis(waveform, ap.sample_rate)
    return waveform

def vocoder_vocoding(vocoder, ap, mel):
    mel = torch.tensor(mel)
    if torch.cuda.is_available():
        mel = mel.cuda()
    mel = mel.unsqueeze(0)
    # Generate
    with torch.no_grad():
        audio = vocoder.inference(mel)
    # Decode
    if ap.vocoder == 'wavernn':
        audio = vocoder_audio.squeeze()
    else:  # 'griffin_lim'
        audio = ap.inv_mel_transform(mel.data.cpu().numpy())
    return audio


tts_model, ap = load_model()
vocoder_model, vocoder_ap = load_vocoder()


text = "سلام، این یک تست برای voice cloning در زبان فارسی است."
waveform = tts(tts_model, ap, text)


output_file = "generated_voice.wav"
librosa.output.write_wav(output_file, waveform.astype(np.float32), ap.sample_rate)

loaded_waveform, sr = librosa.load(output_file, sr=ap.sample_rate)
plt.figure(figsize=(12, 4))
librosa.display.waveshow(loaded_waveform, sr=sr)
plt.title('Generated Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


ipd.Audio(output_file)
