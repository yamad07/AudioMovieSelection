import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(
            self,
            audio_features,
            window_size=2,
            seq_len=10,
            ):

        self.audio_features = audio_features
        self.window_size = window_size
        self.seq_len = seq_len


    def __getitem__(self, idx):

        shot_num = 4
        features = []
        for i in range(self.seq_len):
            features.append(self.audio_features[idx+i:idx+i+self.window_size])

        return torch.Tensor(features)


    def __len__(self):
        return len(self.audio_features) - window_size - seq_len
