import json
import os
import sys
from pathlib import Path


import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visualkeras
from torch import Tensor, from_numpy
from torchaudio.transforms import LFCC
import keras_tuner

import cnn_hypermodel
import resnet
import spectrogram_pipeline
from rnn import build_rnn_model

DEFAULT_SAMPLE_RATE = 22050

LABEL_FAKE = 1
LABEL_REAL = 0

LOSS = "binary_crossentropy"
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 32
DROPOUT_RATIO = 0.30
FEATURE_USED = 'mfcc'
MODEL_USED = 'cnn'
NUMBER_OF_MFCC = 20
NUMBER_OF_LFCC = 40
NUMBER_OF_LAYERS = 3
NUMBER_OF_SECONDS_PER_AUDIO = 5

TRAINING_DATA_PATH = "training"
VALIDATION_DATA_PATH = "validation"
TESTING_DATA_PATH = "testing"
NEW_AUDIO_DATA = "new_test_audio_data"

DATA_AUGMENTATION = True
TRAINING_JSON_PATH = f"norm_training_data_{FEATURE_USED}{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}.json" if DATA_AUGMENTATION is True else f"norm_train_data_{FEATURE_USED}{NUMBER_OF_MFCC}.json"
VALIDATION_JSON_PATH = f"norm_val_data_{FEATURE_USED}{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}.json"
TESTING_JSON_PATH = f"norm_test_data_{FEATURE_USED}{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}.json"
MODEL_PATH = (f"models_for_norm\\model_{MODEL_USED}_{FEATURE_USED}"
              f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}"
              f"_layers{NUMBER_OF_LAYERS if MODEL_USED is not 'resnet' else ''}_lr{LEARNING_RATE}_epochs{EPOCHS}.keras")


PLOT_FOLDER = "plots_for_norm"


# metodi che creano i json con i dati estratti. Solo il training set viene aumentato, per prevenire data leakage
def extract_and_label(data_path, json_path, data_structure, is_training_data,
                      number_of_lfcc=NUMBER_OF_LFCC, number_of_mfcc=NUMBER_OF_MFCC, hop_length=512, n_fft=2048):
    for root, dirs, files in os.walk(data_path):
        if root is not data_path:
            label_name = root.split("\\")[-1]  # l'output è una lista ["training", "fake"] e poi ["training","real"]
            # avendo solo due categorie si poteva magari fare senza impostare questo ciclo, però manteniamo flessibiltà
            print(f"Processing {label_name} audio files")
            data_structure["mappings"].append(label_name)
            for file in files:
                file_path = os.path.join(root, file)
                audio_signal, sample_rate = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)

                label = LABEL_REAL if label_name == 'real' else LABEL_FAKE
                if len(audio_signal) >= DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO:
                    audio_signal = audio_signal[:DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO]
                elif len(audio_signal) < DEFAULT_SAMPLE_RATE:
                    print(f"{file_path} is too small (less 1 sec)")
                    continue
                else:
                    audio_signal = librosa.util.fix_length(audio_signal,
                                                           size=DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO)

                if FEATURE_USED == 'mfcc':
                    mfcc_list = librosa.feature.mfcc(y=audio_signal, n_mfcc=number_of_mfcc,
                                                     n_fft=n_fft, hop_length=hop_length)
                    data_structure["input_feature"].append(mfcc_list.T.tolist())
                    data_structure["labels"].append(label)

                elif FEATURE_USED == 'lfcc':
                    lfcc_transform = LFCC(sample_rate=sample_rate, n_lfcc=number_of_lfcc,
                                          speckwargs={
                                              "n_fft": n_fft, "hop_length": hop_length
                                          })
                    lfccs = lfcc_transform(from_numpy(audio_signal))
                    data_structure['input_feature'].append(Tensor.tolist(lfccs))
                    data_structure["labels"].append(label)

                print(f"{file_path} is labeled {label_name}({label})")

    save_json(json_path, data_structure)


def save_json(json_path, data_structure):
    print(f"Saving {json_path}")
    with open(json_path, "w") as outfile:
        json.dump(data_structure, outfile, indent=4)
        print("data saved")


def prepare_dataset():
    training_data = {
        "mappings": [],
        "labels": [],
        "input_feature": []
    }
    validation_data = {
        "mappings": [],
        "labels": [],
        "input_feature": []
    }
    testing_data = {
        "mappings": [],
        "labels": [],
        "input_feature": []
    }
    extract_and_label(TRAINING_DATA_PATH, TRAINING_JSON_PATH, training_data, True)
    extract_and_label(VALIDATION_DATA_PATH, VALIDATION_JSON_PATH, validation_data, False)
    extract_and_label(TESTING_DATA_PATH, TESTING_JSON_PATH, testing_data, False)


def load_input_and_target_data(json_path):
    print("Loading data from data json")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    print("Data loaded")
    x = np.array(json_data["input_feature"])
    y = np.array(json_data["labels"])
    return x, y


def plot_history_and_save_plot_to_file(history, model_type=MODEL_USED, feature_used=FEATURE_USED, plot_path=None,
                                       learning_rate=LEARNING_RATE, epochs=EPOCHS,):
    fig, axs = plt.subplots(1, 2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    text = f"Learning rate of {learning_rate}, trained for {epochs} epochs.\nFeature used: {feature_used}"
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)

    plot_file_path = plot_path if plot_path is not None else (f"{PLOT_FOLDER}\\{model_type}_lr{learning_rate}_epochs{epochs}_{feature_used}"
                      f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}"
                      f"_layers{NUMBER_OF_LAYERS if FEATURE_USED != 'resnet' else ''}.png")
    plt.savefig(plot_file_path)
    plt.show()


def pick_audio_from_test_folders_and_return_x_and_y_for_testing(feature_used=FEATURE_USED):
    test_data = {
        "label": [],
        "feature_used": []
    }
    fake_files = os.listdir(f"{NEW_AUDIO_DATA}\\fake")
    real_files = os.listdir(f"{NEW_AUDIO_DATA}\\real")

    print("Picking new fake files")
    for file in fake_files:
        audio, sr = librosa.load(f"{NEW_AUDIO_DATA}\\fake\\{file}", sr=DEFAULT_SAMPLE_RATE)
        if len(audio) >= DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO:
            audio = audio[:DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO]
        elif len(audio) < DEFAULT_SAMPLE_RATE:
            print("file too small")
            continue
        else:
            audio = librosa.util.fix_length(audio, size=DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO)

        if feature_used == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, n_mfcc=NUMBER_OF_MFCC, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mfccs.T.tolist())

        elif feature_used == 'lfcc':
            lfcc_transform = LFCC(sample_rate=sr, n_lfcc=NUMBER_OF_LFCC,
                                  speckwargs={
                                      "n_fft": 2048, "hop_length": 512
                                  })
            lfccs = lfcc_transform(from_numpy(audio))
            test_data['feature_used'].append(Tensor.tolist(lfccs))

        test_data["label"].append(LABEL_FAKE)

    print("Picking new real files")
    for file in real_files:
        audio, sr = librosa.load(f"{NEW_AUDIO_DATA}\\real\\{file}", sr=DEFAULT_SAMPLE_RATE)
        if len(audio) >= DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO:
            audio = audio[:DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO]  # dovrebbe restituire 5 secondi di audio.

        elif len(audio) < DEFAULT_SAMPLE_RATE:
            print("file too small")
            continue
        else:
            audio = librosa.util.fix_length(audio, size=DEFAULT_SAMPLE_RATE * NUMBER_OF_SECONDS_PER_AUDIO)
        if feature_used == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, n_mfcc=NUMBER_OF_MFCC, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mfccs.T.tolist())

        elif feature_used == 'lfcc':
            lfcc_transform = LFCC(sample_rate=sr, n_lfcc=NUMBER_OF_LFCC,
                                  speckwargs={
                                      "n_fft": 2048, "hop_length": 512
                                  })
            lfccs = lfcc_transform(from_numpy(audio))
            test_data['feature_used'].append(Tensor.tolist(lfccs))

        test_data["label"].append(LABEL_REAL)

    test_inputs = np.array(test_data["feature_used"])
    print(f"{test_inputs} ; {test_inputs.shape}")
    test_targets = np.array(test_data["label"])
    print(f"{test_targets} ; {test_targets.shape}")
    print("New test data created")
    return test_inputs, test_targets


def cnn_pipeline_from_tuner_to_test(input_shape, X_train, X_test, y_train, y_test, X_val, y_val):
    my_hypermodel = cnn_hypermodel.MyCnnHyperModel(input_shape, NUMBER_OF_LAYERS, DROPOUT_RATIO, LEARNING_RATE)
    model_filepath = MODEL_PATH
    tuner = keras_tuner.RandomSearch(
        my_hypermodel,
        objective='val_loss',
        overwrite=True,
        max_trials=5)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    print("Beginning tuning process")
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[stop_early])

    hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters found were: ", hp.values)
    hypermodel = tuner.hypermodel.build(hp)
    history = hypermodel.fit(X_train, y_train, epochs=EPOCHS,
                             validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                        tf.keras.callbacks.ModelCheckpoint(
                                            filepath=model_filepath,
                                            monitor='val_loss', mode='min', save_best_only=True
                                        )])

    test_loss, test_acc, test_precision, test_recall = hypermodel.evaluate(X_test, y_test)
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print("\nTest loss: {}, test accuracy: {}, test precision: {}, test recall: {}".
          format(test_loss, test_acc, test_precision, test_recall))
    print("F1 score: {}".format(f1_score))
    return history, test_loss, test_acc, model_filepath


def plot_new_test_results_and_compare_to_old_evaluate(old_test_loss, new_test_loss, old_test_acc, new_test_acc,
                                                      model_type=MODEL_USED, feature_used=FEATURE_USED):
    fig, axs = plt.subplots(1, 2)
    ax_0_bar = axs[0].bar(['old loss', 'new loss'], [old_test_loss, new_test_loss],
                          label=['old loss', 'new loss'], color=['tab:green', 'tab:red'])
    axs[0].set_ylabel('Loss comparison')
    axs[0].bar_label(ax_0_bar)

    ax_1_bar = axs[1].bar(['old acc', 'new acc'], [old_test_acc, new_test_acc],
                          label=['old acc', 'new acc'], color=['tab:green', 'tab:red'])
    axs[1].set_ylabel('Accuracy comparison')
    axs[1].bar_label(ax_1_bar)

    text = (f"Loss and accuracy comparison of model on audio_data vs unseen new audio from same dataset\n"
            f"Feature used: {feature_used}")
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=10)
    plot_file_path = (f"{PLOT_FOLDER}\\new_tests_{model_type}_lr{LEARNING_RATE}_epochs{EPOCHS}_{feature_used}"
                      f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else NUMBER_OF_LFCC if FEATURE_USED == 'lfcc' else ''}.png"
                      f"_layers{NUMBER_OF_LAYERS}.png")
    plt.savefig(plot_file_path)
    plt.show()


def plot_loss_and_accuracy(loss, acc):
    fig, axs = plt.subplots()
    ax_0_bar = axs.bar(['loss', 'accuracy'], [loss, acc],
                       label=['loss', 'accuracy'])
    axs.set_ylabel('Loss and accuracy')
    axs.bar_label(ax_0_bar)

    text = f"Loss and accuracy of model on unseen data"
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=10)
    plot_file_path = (
        f"{PLOT_FOLDER}\\new_data_predictions_{MODEL_USED}_lr{LEARNING_RATE}_epochs{EPOCHS}_{FEATURE_USED}"
        f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else ''}.png"
        f"_layers{NUMBER_OF_LAYERS}.png")
    plt.savefig(plot_file_path)
    plt.show()


def main():
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    visualkeras.layered_view(model, to_file='architectures/cnnmodel.png', legend=True).show()
    sys.exit()
    """
    """
    if FEATURE_USED == 'spectrogram':
        spectrogram_pipeline.main()
        sys.exit()
    """

    test_loss = None
    test_acc = None
    data_path = Path(TRAINING_JSON_PATH)
    if not data_path.exists():
        prepare_dataset()

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        x_train, y_train = load_input_and_target_data(TRAINING_JSON_PATH)
        x_val, y_val = load_input_and_target_data(VALIDATION_JSON_PATH)
        x_test, y_test = load_input_and_target_data(TESTING_JSON_PATH)

        # (num_segmenti_audio, 13 coefficienti, num canali(che è uno))
        print(x_train.shape)
        if 'lstm' in MODEL_USED:
            input_shape = (x_train.shape[1], x_train.shape[2])
        else:
            input_shape = (x_train.shape[1], x_train.shape[2], 1)

        # codice duplicato
        if MODEL_USED == 'resnet':
            res = resnet.ResNetModel(input_shape=input_shape, learning_rate=LEARNING_RATE)
            resnet_model = res.build_res_network()
            history = resnet_model.fit(x_train, y_train, epochs=EPOCHS,
                                       validation_data=(x_val, y_val), batch_size=BATCH_SIZE,
                                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                                  tf.keras.callbacks.ModelCheckpoint(
                                                      filepath=MODEL_PATH,
                                                      monitor='val_loss', mode='min', save_best_only=True
                                                  )])
            test_loss, test_acc = resnet_model.evaluate(x_test, y_test)
            plot_history_and_save_plot_to_file(history)
        elif 'lstm' in MODEL_USED:
            lstm_model = build_rnn_model(MODEL_USED, input_shape, LEARNING_RATE)
            history = lstm_model.fit(x_train, y_train, epochs=EPOCHS,
                                     validation_data=(x_val, y_val), batch_size=BATCH_SIZE,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                                tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=MODEL_PATH,
                                                    monitor='val_loss', mode='min', save_best_only=True
                                                )])
            test_loss, test_acc = lstm_model.evaluate(x_test, y_test)
            plot_history_and_save_plot_to_file(history)
        else:
            history, test_loss, test_acc, model_filepath = cnn_pipeline_from_tuner_to_test(input_shape, x_train, x_test,
                                                                                           y_train, y_test, x_val,
                                                                                           y_val)
            plot_history_and_save_plot_to_file(history)

    # un altro giro di evaluate su dati mai visti. Prelevati 300 di ognuno e etichettati per testare il modello
    """
    test_inputs, test_targets = pick_audio_from_test_folders_and_return_x_and_y_for_testing(FEATURE_USED)
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    new_test_loss, new_test_acc, new_ = model.evaluate(test_inputs, test_targets)
    print("\nLoss su nuovi dati: {}, Accuracy su nuovi dati: {}".format(new_test_loss, new_test_acc))
    plot_loss_and_accuracy(new_test_loss, new_test_acc)
    if test_loss is not None and test_acc is not None:
        plot_new_test_results_and_compare_to_old_evaluate(test_loss, new_test_loss, test_acc, new_test_acc)
    """

if __name__ == "__main__":
    main()
