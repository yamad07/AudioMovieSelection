import numpy as np
import librosa
from datetime import datetime
import subprocess
import soundfile as sf
from pydub import AudioSegment
import os


class AudioFeaturePreproccessing:
    @classmethod
    def from_mp4(
            cls,
            mp4_file_name
    ):
        call_list  = ['ffmpeg']
        call_list += [
            '-i',
            '{}.mp4'.format(mp4_file_name),
            '-c',
            'copy']
        call_list += ['-map', '0']
        call_list += ['-segment_time', '5']
        call_list += ['-f', 'segment']
        call_list += ['outputs/%03d.mp4']

        subprocess.call(call_list)
        listdir = os.listdir('outputs')
        freqs = []
        for mp4_file_name in listdir:
            data, fs = librosa.core.load('outputs/{}'.format(mp4_file_name), sr=16000)
            # data, fs = sf.read(bytes_io)
            k = 3  # sample episode num
            time_unit = 3  # unit: second
            mean = (data.max() + data.min()) / 2
            span = (data.max() - data.min()) / 2
            if span < 1e-6:
                span = 1
            data = (data - mean) / span  # range: [-1,1]

            D = librosa.core.stft(data, n_fft=512)
            freq = np.abs(D)
            freq = librosa.core.amplitude_to_db(freq)

            # tile
            rate = freq.shape[1] / (len(data) / fs)
            thr = int(np.ceil(time_unit * rate / k * (k + 1)))
            copy_ = freq.copy()
            while freq.shape[1] < thr:
                tmp = copy_.copy()
                freq = np.concatenate((freq, tmp), axis=1)
            # sample
            n = freq.shape[1]
            milestone = [x[0] for x in np.array_split(np.arange(n), k + 1)[1:]]
            span = 15
            stft_img = []
            for i in range(k):
                stft_img.append(freq[:, milestone[i] - span:milestone[i] + span])
            freq = np.concatenate(stft_img, axis=1)
            freqs.append(freq)

        freqs = np.array(freqs)
        return freqs
