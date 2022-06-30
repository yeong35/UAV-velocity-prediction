import os
import librosa
import soundfile as sf
import numpy as np

def trim_audio_data(audio_file, save_file, start_sec=0, end_sec=10):

    y, sr = librosa.load(audio_file)

    ny = y[start_sec*sr:end_sec*sr]

    sf.write(save_file + '.wav', ny, sr)

base_path = './dataset'

audio_path = base_path + '/big_slow'
save_path = base_path + '/test'

audio_list = os.listdir(audio_path)

for audio_name in audio_list:
    if audio_name.find('wav') is not -1:
        audio_file = audio_path + '/' + audio_name
        save_file = save_path + '/' + audio_name[:-4]

        trim_audio_data(audio_file, save_file)