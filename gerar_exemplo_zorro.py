import os
import numpy as np
import scipy.io.wavfile as wavfile


# Generate a synthetic dataset
def generate_synth_keystroke(freq, duration=0.1, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple decaying sine wave + noise for "keystroke" sound
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 30)
    noise = np.random.randn(len(t)) * 0.1 * np.exp(-t * 20)
    return (tone + noise) * 32767


def create_dataset():
    os.makedirs("dados_falsos", exist_ok=True)
    letters = "ozreigs "
    freqs = np.linspace(300, 1000, len(letters))

    for i, char in enumerate(letters):
        char_dir = os.path.join("dados_falsos", "espaco" if char == " " else char)
        os.makedirs(char_dir, exist_ok=True)
        for j in range(5):  # 5 examples per char
            audio = generate_synth_keystroke(freqs[i])
            wavfile.write(
                os.path.join(char_dir, f"{j}.wav"), 44100, audio.astype(np.int16)
            )


def create_test_sentence():
    # "o zorro e gris"
    sentence = "o zorro e gris"
    letters = "ozreigs "
    freqs = dict(zip(letters, np.linspace(300, 1000, len(letters))))

    sr = 44100
    audio_full = np.zeros(0)
    for char in sentence:
        ks = generate_synth_keystroke(freqs[char])
        pause = np.zeros(int(sr * 0.2))  # 200ms pause between keys
        audio_full = np.concatenate([audio_full, ks, pause])

    wavfile.write("o_zorro_e_gris.wav", sr, audio_full.astype(np.int16))


if __name__ == "__main__":
    create_dataset()
    create_test_sentence()
    print("Dataset e áudio de teste criados!")
    print("Execute: poetry run python hacker_de_teclado.py --treinar dados_falsos")
    print("Em seguida: poetry run python hacker_de_teclado.py --prever o_zorro_e_gris.wav")
