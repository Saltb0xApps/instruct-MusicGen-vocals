import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import json

class CassetteDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('-beat.json'):
                base_name = file_name.replace('-beat.json', '')
                instrumental_path = os.path.join(self.data_dir, f"{base_name}-beat.m4a")
                vocal_path = os.path.join(self.data_dir, f"{base_name}.m4a")
                with open(os.path.join(self.data_dir, f"{base_name}.json"), 'r') as f:
                    lyrics_data = json.load(f)
                instruction = "instruct: add vocal stem."
                lyrics = lyrics_data['lyrics']  # Adjust this line based on the actual JSON structure
                data.append((instrumental_path, vocal_path, instruction, lyrics))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instrumental_path, vocal_path, instruction, lyrics = self.data[idx]
        instrumental = self.load_audio(instrumental_path)
        vocal = self.load_audio(vocal_path)

        if self.transform:
            instrumental = self.transform(instrumental)
            vocal = self.transform(vocal)

        return instrumental, vocal, instruction, lyrics

    def load_audio(self, file_path):
        audio, _ = sf.read(file_path)
        return torch.tensor(audio, dtype=torch.float32)

class CassetteDataModule:
    def __init__(self, data_dir, batch_size, num_workers=4, pin_memory=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.data_train = CassetteDataset(os.path.join(self.data_dir, 'train'))
        self.data_val = CassetteDataset(os.path.join(self.data_dir, 'val'))
        self.data_test = CassetteDataset(os.path.join(self.data_dir, 'test'))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
