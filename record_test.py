import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time


def record_phrase(filename="meu_teste_zorro.wav", duration=8, sr=44100):
    print("======================================")
    print("🎤 RECORDING YOUR ATTACK PHRASE")
    print("======================================")
    print("When the recording starts, type the phrase: 'o zorro e gris'")
    input("Press ENTER when you are ready to start...")

    print("\nRecording starting in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print(f"\n🔴 RECORDING for {duration} seconds! TYPE NOW!")

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished

    print("\n✅ Recording finished.")

    # Convert to 16-bit PCM and save
    y_int16 = np.int16(recording.flatten() * 32767)
    wavfile.write(filename, sr, y_int16)

    print(f"✅ Saved your typing audio to '{filename}'!")


if __name__ == "__main__":
    record_phrase()
