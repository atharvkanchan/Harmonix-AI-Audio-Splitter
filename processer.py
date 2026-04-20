import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

class AudioSplitter:
    def __init__(self):
        # Loads the Hybrid Transformer Demucs model (v4)
        self.model = get_model('htdemucs')
        self.model.eval()

    def split(self, audio_path):
        # 1. Load the track
        wav, sr = torchaudio.load(audio_path)
        
        # 2. Normalize and Prepare
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        
        # 3. Apply the model
        with torch.no_grad():
            sources = apply_model(self.model, wav[None])[0]
            
        # Returns a dictionary of the 4 isolated tracks
        return {
            "drums": sources[0],
            "bass": sources[1],
            "other": sources[2],
            "vocals": sources[3]
        }, self.model.samplerate
