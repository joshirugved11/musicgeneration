from src.data_preprocessing import preprocess_dataset
from src.model_training import train_model
from src.music_generation import generate_music
import torch
import os
import numpy as np

# Define paths for data and model storage
dataset_path = r"D:\Music Generation\data\raw\GTZAN dataset\Data\genres_original"  # Path to raw dataset
processed_data_dir = r"D:\Music Generation\data\processed"  # Directory to store processed files
models_dir = r"D:\Music Generation\models"  # Directory to save the trained model
generated_music_path = r"D:\Music Generation\GeneratedMusic\generated_music.wav"  # Path to save generated music

# Step 1: Preprocess Data (Resample, Normalize, Pad/Trim)
preprocess_dataset(dataset_path, processed_data_dir)  # Process the whole dataset and save it as .npy files
print(f"Processed data saved at: {processed_data_dir}")

# Step 2: Train Model (Load processed data, Train GPT-2 model)
trained_model = train_model(processed_data_dir, epochs=10, sequence_length=100, batch_size=32)
model_save_path = os.path.join(models_dir, "music_model.pth")
torch.save(trained_model.state_dict(), model_save_path)  # Save the trained model
print(f"Model saved at: {model_save_path}")

# Step 3: Generate New Music using a Random Seed
# Generate music from a random seed sequence (e.g., length 100)
seed_sequence = np.random.uniform(-1, 1, 100)  # Initial random seed of length 100

# Generate music from the trained model
generate_music(seed_sequence, num_samples=16000, sequence_length=100)  # Generate 1 second of audio (16000 samples at 16 kHz)
print(f"Generated music saved at: {generated_music_path}")
