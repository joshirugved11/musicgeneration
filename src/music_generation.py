import torch
from pydub import AudioSegment
import numpy as np
from model_training import SimpleMusicGenerator  # Ensure this import points to your actual model file

# Load the trained model
model_path = "../models/music_model.pth"
input_size = 1  # Single amplitude feature
hidden_size = 128
sequence_length = 100

model = SimpleMusicGenerator(input_size, hidden_size, input_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to generate music using the trained PyTorch model
def generate_music(seed_sequence, num_samples=16000):
    model.eval()
    generated_audio = seed_sequence.tolist()
    current_sequence = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        for _ in range(num_samples):
            prediction = model(current_sequence)
            next_sample = prediction.squeeze().item()
            generated_audio.append(next_sample)
            current_sequence = torch.tensor(generated_audio[-sequence_length:], dtype=torch.float32).unsqueeze(0)

    # Convert generated audio to a numpy array
    generated_audio = np.array(generated_audio, dtype=np.float32)
    generated_audio = (generated_audio * 32768).astype(np.int16)  # Denormalize for saving as audio

    # Save generated audio using pydub
    audio_segment = AudioSegment(
        generated_audio.tobytes(), frame_rate=22050, sample_width=2, channels=1
    )
    audio_segment.export("../data/processed/generated_music.wav", format="wav")
    print("Generated music saved!")

# Example usage: Provide a random seed sequence for generation
seed_sequence = np.random.uniform(-1, 1, sequence_length)
generate_music(seed_sequence)
