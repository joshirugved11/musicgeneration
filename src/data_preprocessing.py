from pydub import AudioSegment
import numpy as np
import os
from pathlib import Path

def preprocess_audio(file_path, target_sr=22050, target_duration=30):
    """
    Preprocess an audio file using pydub.
    - Resample to `target_sr`
    - Normalize audio
    - Trim or pad to `target_duration` seconds
    """
    try:
        audio = AudioSegment.from_file(file_path)

        # Resample if necessary
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        # Convert to mono
        audio = audio.set_channels(1)

        # Normalize audio (scale samples to [-1, 1])
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        # Calculate the target length in samples
        target_length = target_sr * target_duration

        # Trim or pad the audio
        current_length = len(audio_array)
        if current_length > target_length:
            audio_array = audio_array[:target_length]
        else:
            padding = target_length - current_length
            audio_array = np.pad(audio_array, (0, padding), 'constant')

        return audio_array

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_dataset(dataset_path, output_path):
    """Preprocess all audio files in a dataset folder."""
    os.makedirs(output_path, exist_ok=True)
    for genre_folder in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre_folder)
        output_genre_path = os.path.join(output_path, genre_folder)
        os.makedirs(output_genre_path, exist_ok=True)

        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if Path(file).suffix in {'.wav', '.mp3'}:
                    file_path = os.path.join(genre_path, file)
                    processed_audio = preprocess_audio(file_path)

                    if processed_audio is not None:
                        output_file_path = os.path.join(output_genre_path, f"{Path(file).stem}.npy")
                        np.save(output_file_path, processed_audio)
                        print(f"Processed and saved: {output_file_path}")

# Example usage
dataset_path = r"D:\Music Generation\data\raw\GTZAN dataset\Data\genres_original"
output_path = r"D:\Music Generation\data\processed"
preprocess_dataset(dataset_path, output_path)
