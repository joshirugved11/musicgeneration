import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader

class SimpleMusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Taking the last time step output
        return out

class MusicDataset(Dataset):
    """Custom dataset for loading .npy files for music generation."""
    def __init__(self, base_dir, sequence_length=100):
        self.sequence_length = sequence_length
        self.files = glob.glob(os.path.join(base_dir, '*', '*.npy'))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        
        # Normalize audio data
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Convert to tensor and create sequences
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(-1)  # Shape [N, 1]
        
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Data length too short for sequence length in {file_path}")
        
        # Random start index for sequence selection
        start_idx = np.random.randint(0, len(data) - self.sequence_length - 1)
        sequence = data[start_idx:start_idx + self.sequence_length]
        target = data[start_idx + self.sequence_length]
        
        return sequence, target

def train_model(base_dir, epochs=10, sequence_length=100, batch_size=32):
    """Train the LSTM model using a DataLoader for multiple files."""
    dataset = MusicDataset(base_dir, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = 1  # Single amplitude feature per time step
    model = SimpleMusicGenerator(input_size, hidden_size=128, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_data, batch_targets in dataloader:
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}")

    torch.save(model.state_dict(), "../models/music_model.pth")
    print("Model saved!")
    return model

# Example usage:
base_dir = r"D:\Music Generation\data\processed"
trained_model = train_model(base_dir)
