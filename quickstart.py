import os
import time
import argparse
import subprocess
import glob
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

try:
    import sounddevice as sd
except ImportError:
    print("Please make sure 'sounddevice' is installed (e.g., poetry add sounddevice)")
    exit(1)


def record_key(key_name, duration=5, sr=44100, output_dir="dados_treino"):
    print(f"\n======================================")
    print(f"Preparing to record key: [{key_name}]")
    print(f"======================================")
    print(f"When the recording starts, press the '{key_name}' key repeatedly.")
    print(f"Leave a small pause (e.g., ~0.5s) between each press.")
    input("Press ENTER when you are ready to start recording...")

    print("\nRecording starting in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print(f"🔴 RECORDING for {duration} seconds! PRESS '{key_name}' REPEATEDLY!")

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()  # Wait until recording is finished

    print("✅ Recording finished.")

    # Flatten array
    y = recording.flatten()

    print("⏳ Detecting keystrokes with librosa (this might take a few seconds)...")
    # We will use librosa to detect onsets (keystrokes)
    try:
        onsets_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onsets = librosa.frames_to_samples(onsets_frames)
    except Exception as e:
        print(f"Error during onset detection: {e}")
        return 0

    print(f"🔍 Found {len(onsets)} potential keystrokes.")

    if len(onsets) == 0:
        print(f"⚠️ No keystrokes detected for '{key_name}'! Try hitting the key harder.")
        return 0

    # Save the full recording so hacker_de_teclado.py can extract the keystrokes
    key_dir = os.path.join(output_dir, key_name)
    os.makedirs(key_dir, exist_ok=True)

    # Try not to overwrite existing files if appending
    existing_files = glob.glob(os.path.join(key_dir, "*.wav"))
    start_idx = len(existing_files) + 1

    out_file = os.path.join(key_dir, f"{start_idx}.wav")

    # Convert to 16-bit PCM for standard wav
    y_int16 = np.int16(y / np.max(np.abs(y)) * 32767)
    wavfile.write(out_file, sr, y_int16)

    print(
        f"✅ Saved full {duration}s recording for key '{key_name}' with {len(onsets)} keystrokes."
    )
    return len(onsets)


def main():
    parser = argparse.ArgumentParser(description="Quickstart Prompter for Hacker de Teclado")
    parser.add_argument(
        "--dir", default="dados_treino", help="Output directory for training data"
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Welcome to Hacker de Teclado Quickstart Engine ⌨️🎧")
    print("=" * 50)

    keys_input = input(
        "\nEnter the keys you want to record, separated by commas\n[Default: 1-0, a-z] (Press ENTER for default): "
    )

    if not keys_input.strip():
        # Default: 1-0 and a-z
        keys_list = list("1234567890abcdefghijklmnopqrstuvwxyz")
    else:
        keys_list = [k.strip() for k in keys_input.split(",") if k.strip()]

    if not keys_list:
        print("No keys specified. Exiting.")
        return

    total_samples = 0
    for key in keys_list:
        samples = record_key(key, duration=5, output_dir=args.dir)
        total_samples += samples

    print(f"\n======================================")
    print(f"Recording Phase Complete!")
    print(f"Total samples collected across all keys: {total_samples}")
    print(f"Data saved to: {args.dir}/")
    print(f"======================================")

    if total_samples > 0:
        ans = input(f"\nDo you want to start training the model now? (y/n): ")
        if ans.lower() == "y":
            print("\n🚀 Starting training process...\n")
            subprocess.run(
                ["poetry", "run", "python", "hacker_de_teclado.py", "--treinar", args.dir]
            )
            print(
                "\n🎉 Training completed! You can now use the model with the --prever argument."
            )
        else:
            print("\nSkipping training. You can train later by running:")
            print(f"poetry run python hacker_de_teclado.py --treinar {args.dir}")
    else:
        print("\nNo samples were recorded successfully. Please try again.")


if __name__ == "__main__":
    main()
