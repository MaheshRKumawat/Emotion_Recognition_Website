import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def wave_plot(data, sr, emotion, color):
    plt.figure(figsize=(12, 5))
    plt.title(f'{emotion} emotion for waveplot', size=17)
    librosa.display.waveshow(y=data, sr=sr, color=color)


def spectogram(data, sr, emotion):
    audio = librosa.stft(data)
    audio_db = librosa.amplitude_to_db(abs(audio))
    plt.figure(figsize=(12, 5))
    plt.title(f'{emotion} emotion for spectogram', size=17)
    librosa.display.specshow(audio_db, sr=sr, x_axis='time', y_axis='hz')


def add_noise(data, random=False, rate=0.035, threshold=0.075):
    if random:
        rate = np.random.random()*threshold
    noise = rate*np.random.uniform()*np.amax(data)
    augmented_data = data+noise*np.random.normal(size=data.shape[0])
    return augmented_data


def shifting(data, rate=1000):
    augmented_data = int(np.random.uniform(low=-5, high=5)*rate)
    augmented_data = np.roll(data, augmented_data)
    return augmented_data


def pitching(data, sr, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(data, sr, pitch_factor)


def streching(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(
        data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(
        data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(data, sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)


def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    aud = extract_features(data, sr)
    audio = np.array(aud)

    noised_audio = add_noise(data, random=True)
    aud2 = extract_features(noised_audio, sr)
    audio = np.vstack((audio, aud2))

    pitched_audio = pitching(data, sr, random=True)
    aud3 = extract_features(pitched_audio, sr)
    audio = np.vstack((audio, aud3))

    pitched_audio1 = pitching(data, sr, random=True)
    pitched_noised_audio = add_noise(pitched_audio1, random=True)
    aud4 = extract_features(pitched_noised_audio, sr)
    audio = np.vstack((audio, aud4))

    return audio
