import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def main():
    audio_file = "training\\real\\file329.wav_16k.wav_norm.wav_mono.wav_silence.wav"
    audio, sr = librosa.load(audio_file, sr=16000)
    audio = audio[:16000 * 5]
    mfcc_list = librosa.feature.mfcc(y=audio, n_mfcc=20,
                                     n_fft=2048, hop_length=512)
    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(mfcc_list.T.tolist(), 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    ax.set_title('MFCC')

    plt.show()

if __name__ == '__main__':
    main()