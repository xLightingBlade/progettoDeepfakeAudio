import json
import os
import random
from pathlib import Path
import resnet

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_tuner
from keras_tuner import HyperModel

DATASET_PATH = "audio_data"
TEST_AUDIO_PATH = "test_audio_data"
JSON_PATH = "data.json"
DEFAULT_SAMPLE_RATE = 22050

LABEL_FAKE = 1
LABEL_REAL = 0

LOSS = "binary_crossentropy"
LEARNING_RATE = 0.0001
EPOCHS = 25
BATCH_SIZE = 32
DROPOUT_RATIO = 0.25
FEATURE_USED = 'mel_spectrogram'
MODEL_USED = 'cnn'
NUMBER_OF_MFCC = 20
NUMBER_OF_LAYERS = 1

data = {
    "mappings": [],
    "labels": [],
    "mfcc": [],
    "mel_spectrogram": [],
}


class MyCnnHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=self.input_shape))
        # reg. l2 assieme al dropout: studi mostrano come porta maggior riduzione della loss
        model.add(tf.keras.layers.Conv2D(filters=hp.Int('first_conv_filters', min_value=32, max_value=128, step=16),
                                         kernel_size=hp.Choice('first_conv_kernel', values=[2, 5]), activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        if NUMBER_OF_LAYERS > 1:
            model.add(tf.keras.layers.Conv2D(filters=hp.Int('second_conv_filters', min_value=16, max_value=64, step=16),
                                             kernel_size=hp.Choice('second_conv_kernel', values=[2, 5]),
                                             activation='relu',
                                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            if NUMBER_OF_LAYERS > 2:
                model.add(
                    tf.keras.layers.Conv2D(filters=hp.Int('third_conv_filters', min_value=16, max_value=64, step=16),
                                           kernel_size=hp.Choice('third_conv_kernel', values=[2, 5]),
                                           activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16),
                                        activation='relu'))
        if hp.Boolean("dropout"):
            model.add(tf.keras.layers.Dropout(DROPOUT_RATIO))

        # output layer, funzione di attivazione sigmoide per class. binaria, 1 neurone per input feature, quindi uno
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=LOSS,
                      metrics=["accuracy"])
        model.summary()

        return model


def prepare_dataset(dataset_path, json_path, number_of_mfcc=NUMBER_OF_MFCC, hop_length=512, n_fft=2048):
    for root, dirs, files in os.walk(dataset_path):
        if root is not dataset_path:
            label_name = root.split("\\")[-1]  # l'output è una lista ["training", "fake"] e poi ["training","real"]
            # avendo solo due categorie si poteva magari fare senza impostare questo ciclo, però manteniamo flessibiltà
            print(f"Processing {label_name} audio files")
            data["mappings"].append(label_name)
            for file in files:
                file_path = os.path.join(root, file)
                print(f"file path : {file_path}")
                audio_signal, sample_rate = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)
                # i file audio di questo dataset sono tutti troncati a 2 secondi quindi hanno la stessa forma

                # coefficienti (mfcc) della traccia audio
                mfcc_list = librosa.feature.mfcc(y=audio_signal, n_mfcc=number_of_mfcc,
                                                 n_fft=n_fft, hop_length=hop_length)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=DEFAULT_SAMPLE_RATE,
                                                                 n_fft=n_fft, hop_length=hop_length)

                label = LABEL_REAL if label_name == 'real' else LABEL_FAKE
                data["labels"].append(label)
                # librosa restituisce gli mfcc come un array 2d, noi vorremmo 1d quindi facciamo la trasposta
                data["mfcc"].append(mfcc_list.T.tolist())
                # provo ad estrarre anche gli spettrogrammi mel-scaled
                data["mel_spectrogram"].append(mel_spectrogram.T.tolist())
                print(f"{file_path} is labeled {label_name}({label})")

    print("Data labeling completed")
    with open(json_path, "w") as f:
        print("Saving data in json")
        json.dump(data, f, indent=4)
        print("Data saved")

    return data


def load_input_and_target_data(json_path, input_feature_to_use=FEATURE_USED):
    print("Loading data from data.json")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    print("Data loaded")
    x = np.array(json_data["mfcc"]) if input_feature_to_use == 'mfcc' else np.array(json_data["mel_spectrogram"])
    y = np.array(json_data["labels"])
    return x, y


def plot_history_and_save_plot_to_file(history, model_type=MODEL_USED, feature_used=FEATURE_USED):
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

    text = f"Learning rate of {LEARNING_RATE}, trained for {EPOCHS} epochs.\nFeature used: {feature_used}"
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
    plot_file_path = (f"plots\\{model_type}_lr{LEARNING_RATE}_epochs{EPOCHS}_{feature_used}"
                      f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else ''}.png"
                      f"_layers{NUMBER_OF_LAYERS}.png")
    plt.savefig(plot_file_path)
    plt.show()


def pick_random_audio_from_test_folders_and_return_x_and_y_for_testing(feature_used=FEATURE_USED,
                                                                       number_of_files_picked=1000):
    test_data = {
        "label": [],
        "feature_used": []
    }
    fake_files = os.listdir(f"{TEST_AUDIO_PATH}\\fake")
    real_files = os.listdir(f"{TEST_AUDIO_PATH}\\real")
    print("Picking new fake files")
    for j in range(number_of_files_picked):
        random_file = random.choice(fake_files)
        audio, sr = librosa.load(f"{TEST_AUDIO_PATH}\\fake\\{random_file}", sr=DEFAULT_SAMPLE_RATE)
        if feature_used == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, n_mfcc=NUMBER_OF_MFCC, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mfccs.T.tolist())

        elif feature_used == 'mel_spectrogram':
            mel_spectrograms = librosa.feature.melspectrogram(y=audio, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mel_spectrograms.T.tolist())

        test_data["label"].append(LABEL_FAKE)

    print("Picking new real files")
    for k in range(number_of_files_picked):
        random_file = random.choice(real_files)
        audio, sr = librosa.load(f"{TEST_AUDIO_PATH}\\real\\{random_file}", sr=DEFAULT_SAMPLE_RATE)
        if feature_used == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, n_mfcc=NUMBER_OF_MFCC, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mfccs.T.tolist())

        elif feature_used == 'mel_spectrogram':
            mel_spectrograms = librosa.feature.melspectrogram(y=audio, n_fft=2048, hop_length=512)
            test_data["feature_used"].append(mel_spectrograms.T.tolist())

        test_data["label"].append(LABEL_REAL)

        test_inputs = np.array(test_data["feature_used"])
        test_targets = np.array(test_data["label"])
        print("New test data created")
        return test_inputs, test_targets


def cnn_pipeline_from_tuner_to_test(input_shape, X_train, X_test, y_train, y_test, X_val, y_val):
    my_hypermodel = MyCnnHyperModel(input_shape)
    model_filepath = (f"models\\model_{MODEL_USED}_{FEATURE_USED}{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else ''}"
                      f"_layers{NUMBER_OF_LAYERS}_lr{LEARNING_RATE}.keras")
    tuner = keras_tuner.RandomSearch(
        my_hypermodel,
        objective='val_loss',
        max_trials=5)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    print("Beginning tuning process")
    tuner.search(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[stop_early])

    hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters found were: ", hp.values)
    hypermodel = tuner.hypermodel.build(hp)
    history = hypermodel.fit(X_train, y_train, epochs=EPOCHS,
                             validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
                             callbacks=[tf.keras.callbacks.EarlyStopping(patience=3),
                                        tf.keras.callbacks.ModelCheckpoint(
                                            filepath=model_filepath,
                                            monitor='val_loss', mode='min', save_best_only=True
                                        )])

    test_loss, test_acc = hypermodel.evaluate(X_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, test_acc))
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
    plot_file_path = (f"plots\\new_tests_{model_type}_lr{LEARNING_RATE}_epochs{EPOCHS}_{feature_used}"
                      f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else ''}.png"
                      f"_layers{NUMBER_OF_LAYERS}.png")
    plt.savefig(plot_file_path)
    plt.show()


def check_if_resnet(feature, model):
    return feature == 'mel_spectrogram' and model == 'resnet'


def main():
    # tf.keras.utils.set_random_seed(1337) attivare per ottenere riproducibilità
    data_path = Path(JSON_PATH)
    if not data_path.exists():
        all_data = prepare_dataset(DATASET_PATH, JSON_PATH)  # commentato, già eseguito una volta. TODO refactor

    x, y = load_input_and_target_data(JSON_PATH, FEATURE_USED)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print(X_train.shape)
    # (num_segmenti_audio, 13 coefficienti, num canali(che è uno))
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    # Refactor necessario, codice duplicato
    if check_if_resnet(FEATURE_USED, MODEL_USED):
        res = resnet.ResNetModel(input_shape=input_shape, learning_rate=LEARNING_RATE)
        resnet_model = res.build_res_network()
        model_filepath = (f"models\\model_{MODEL_USED}_lr{LEARNING_RATE}_epochs{EPOCHS}_{FEATURE_USED}"
                          f"{NUMBER_OF_MFCC if FEATURE_USED == 'mfcc' else ''}"
                          f"_layers{NUMBER_OF_LAYERS}.keras")
        history = resnet_model.fit(X_train, y_train, epochs=EPOCHS,
                                   validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
                                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=3),
                                              tf.keras.callbacks.ModelCheckpoint(
                                                  filepath=model_filepath,
                                                  monitor='val_loss', mode='min', save_best_only=True
                                              )])
        test_loss, test_acc = resnet_model.evaluate(X_test, y_test)
    else:
        history, test_loss, test_acc, model_filepath = cnn_pipeline_from_tuner_to_test(input_shape, X_train, X_test,
                                                                                       y_train, y_test, X_val,
                                                                                       y_val)

    # un altro giro di evaluate su dati mai visti. Prelevati 300 di ognuno e etichettati per testare il modello
    test_inputs, test_targets = pick_random_audio_from_test_folders_and_return_x_and_y_for_testing(FEATURE_USED, 1000)
    model = tf.keras.models.load_model(model_filepath)
    new_test_loss, new_test_acc = model.evaluate(test_inputs, test_targets)
    print("\nLoss su nuovi dati: {}, Accuracy su nuovi dati: {}".format(new_test_loss, new_test_acc))
    plot_history_and_save_plot_to_file(history)
    plot_new_test_results_and_compare_to_old_evaluate(test_loss, new_test_loss, test_acc, new_test_acc)
    """
    model = tf.keras.models.load_model("models\\model_cnn_mfcc20_layers1_lr0.001.keras")
    print(json.loads(json.dumps(model.get_config())))
    print(tf.keras.backend.eval(model.optimizer.learning_rate))
    """


if __name__ == "__main__":
    main()
