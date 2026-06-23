import librosa
import numpy as np

def get_onsets(y, sr):
    raw_onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units="samples")
    filtered = []
    min_dist = sr * 0.3  # 300ms
    
    for o in raw_onsets:
        if not filtered:
            filtered.append(o)
        else:
            if o - filtered[-1] > min_dist:
                filtered.append(o)
    return filtered

y, sr = librosa.load('meu_teste_zorro.wav', sr=44100)
onsets = get_onsets(y, sr)
print(f"Number of onsets detected: {len(onsets)}")
