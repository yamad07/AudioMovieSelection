import torch
from pytube import YouTube
from datetime import datetime
import numpy as np
import io
from .preprocessings import AudioFeaturePreproccessing
from .models.lgss import Model
from .dataset import AudioDataset


class Selector:

    def __init__(
            self,
            youtube_url
    ):
        self.youtube_url = youtube_url
        self.model = Model(
            audio_feature_dim=512,
            lstm_hidden_size=512,
            seq_len=10,
            shot_num=4,
        )
        state_dict = torch.load('./weights/audio_lgss.pth')
        self.model.load_state_dict(state_dict['state_dict'])

    def run(self):
        mp4_file_name = 'sample'
        yt = YouTube(self.youtube_url)
        stream = yt.streams.filter(progressive=True, subtype='mp4').all()[0]
        stream.download(filename=mp4_file_name)
        audio_features = AudioFeaturePreproccessing.from_mp4(mp4_file_name)
        dataset = AudioDataset(
                window_size=4,
                seq_len=10,
                audio_features=audio_features
                )

        for aud_feature in dataset:
            out = self.model.extract_feature(
                audio_feature=aud_feature.unsqueeze(0),
            )
            print(out.size())
