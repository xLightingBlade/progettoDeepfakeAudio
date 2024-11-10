import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), int(label)),
        num_parallel_calls=tf.data.AUTOTUNE)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def remove_small_files(data_path, min_kilobytes):
    fakes = os.listdir(f"{data_path}\\fake")
    real = os.listdir(f"{data_path}\\real")

    for file in fakes:
        if os.path.getsize(os.path.join(data_path, "fake", file)) < 1024 * min_kilobytes - 1:
            os.remove(os.path.join(data_path, "fake", file))
    for file in real:
        if os.path.getsize(os.path.join(data_path, "real", file)) < 1024 * min_kilobytes - 1:
            os.remove(os.path.join(data_path, "real", file))


def get_dataset(data_path, min_file_size):
    remove_small_files(data_path, min_kilobytes=min_file_size)
    dataset = tf.keras.utils.audio_dataset_from_directory(directory=data_path, label_mode='binary',
                                                          labels='inferred',
                                                          batch_size=32,
                                                          output_sequence_length=50000)

    dataset = dataset.map(squeeze, tf.data.AUTOTUNE)
    dataset = make_spec_ds(dataset)
    return dataset


def main():
    training_dataset = get_dataset("training", 35)
    testing_dataset = get_dataset("testing", 35)
    validation_dataset = get_dataset("validation", 35)
    """
    for example_spectrograms, example_spect_labels in training_dataset.take(1):
        print(example_spectrograms.shape, example_spect_labels.shape)
        rows = 3
        cols = 3
        n = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

        for i in range(n):
            r = i // cols
            c = i % cols
            ax = axes[r][c]
            plot_spectrogram(example_spectrograms[i].numpy(), ax)
            ax.set_title("fake" if example_spect_labels[i].numpy() == 0 else "real")
    plt.savefig("example_spectrogram.png")
    plt.show()
    """
    return training_dataset, testing_dataset, validation_dataset


if __name__ == '__main__':
    main()
