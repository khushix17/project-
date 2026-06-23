import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('meu_teste_zorro.wav', sr=44100)
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

# find large peaks
threshold = np.max(rms) * 0.1
peaks = []
for i in range(1, len(rms)-1):
    if rms[i] > rms[i-1] and rms[i] > rms[i+1] and rms[i] > threshold:
        # Check refractory
        if len(peaks) == 0 or (i - peaks[-1]) > (sr * 0.1 / 512):
            peaks.append(i)

print(f"Number of distinct keystrokes found: {len(peaks)}")
