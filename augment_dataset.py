import os
import librosa
import soundfile as sf
import numpy as np

DEFAULT_SAMPLE_RATE = 22050


def augment_dataset(data_path):
    for root, dirs, files in os.walk(data_path):
        if root is not data_path:
            label_name = root.split("\\")[-1]  # l'output è una lista ["training", "fake"] e poi ["training","real"]
            print(f"processing {label_name}")
            for file in files:
                file_path = os.path.join(root, file)
                audio_signal, sample_rate = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)
                gaussian_noise = np.random.normal(0, audio_signal.std(), audio_signal.size)
                noisy_audio = (audio_signal + gaussian_noise).astype(np.float32)
                pitch_shifted_audio = (librosa.effects.pitch_shift(audio_signal, sr=sample_rate,
                                                                   n_steps=4)).astype(np.float32)  #4 semitoni in su

                list = file_path.split("\\")[-1].split(".")[:-1]
                original_file_name = '.'.join(list)
                noisy_audio_filepath = (f"augmented_audio_files\\{label_name}\\augmented_added_noise_"
                                        f"{original_file_name}.wav")
                pitch_shifted_audio_filepath = (f"augmented_audio_files\\{label_name}_augmented_pitch_shifted_"
                                                f"{original_file_name}.wav")

                sf.write(noisy_audio_filepath, noisy_audio, DEFAULT_SAMPLE_RATE)
                # sf.write(pitch_shifted_audio_filepath, pitch_shifted_audio, DEFAULT_SAMPLE_RATE)
