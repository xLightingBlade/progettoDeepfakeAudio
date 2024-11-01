import keras_tuner

import cnn_hypermodel
import get_spectrogram_datasets
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

import resnet

LEARNING_RATE = 0.0001
EPOCHS = 200
LAYERS = 3
MODEL_USED = 'resnet'
MODEL_PATH = (f"models_for_norm\\model_{MODEL_USED}_{LAYERS if MODEL_USED =='cnn' else ''}_spectrogram_tf_data_api_lr0"
              f".0001_epochs200.keras")
PLOT_PATH = f"plots_for_norm\\{MODEL_USED}_lr0.0001__epochs200_spectrogram.png"


def plot_history_and_save_plot_to_file(history, plot_path=PLOT_PATH,
                                       learning_rate=LEARNING_RATE, epochs=EPOCHS):
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

    text = f"Learning rate of {learning_rate}, trained for {epochs} epochs.\nFeature used: spectrogram"
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(plot_path)
    plt.show()


def main():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        training_dataset, testing_dataset, validation_dataset = get_spectrogram_datasets.main()
        training_dataset = training_dataset.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
        testing_dataset = testing_dataset.cache().prefetch(tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)
        for example_spectrograms, example_spect_labels in training_dataset.take(1):
            input_shape = example_spectrograms.shape[1:]  # perchè il primo è il numero di esempi per batch, non serve
        if MODEL_USED == 'resnet':
            resnet_model = resnet.ResNetModel(input_shape=input_shape, learning_rate=LEARNING_RATE)
            res_model = resnet_model.build_res_network()
            history = res_model.fit(training_dataset, epochs=EPOCHS,
                                    validation_data=validation_dataset,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                               tf.keras.callbacks.ModelCheckpoint(
                                                   filepath=MODEL_PATH,
                                                   monitor='val_loss', mode='min', save_best_only=True
                                               )])
            test_loss, test_acc, test_precision,test_recall = res_model.evaluate(testing_dataset)
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        elif MODEL_USED == 'cnn':
            cnn = cnn_hypermodel.MyCnnHyperModel(input_shape, LAYERS, 0.3, LEARNING_RATE)
            tuner = keras_tuner.RandomSearch(
                cnn,
                objective='val_accuracy',
                overwrite=True,
                max_trials=4)
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            print("Beginning tuning process")
            tuner.search(training_dataset, epochs=EPOCHS, validation_data=validation_dataset, callbacks=[stop_early])

            hp = tuner.get_best_hyperparameters()[0]
            print("Best hyperparameters found were: ", hp.values)
            hypermodel = tuner.hypermodel.build(hp)
            history = hypermodel.fit(training_dataset, epochs=EPOCHS,
                                     validation_data=validation_dataset, batch_size=32,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                                                tf.keras.callbacks.ModelCheckpoint(
                                                    filepath=MODEL_PATH,
                                                    monitor='val_loss', mode='min', save_best_only=True
                                                )])

            test_loss, test_acc, test_precision, test_recall = hypermodel.evaluate(testing_dataset)
            f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print("\nLoss dati test: {}, Accuracy dati test: {}, Precision: {}, Recall: {}"
              .format(test_loss, test_acc, test_precision, test_recall))
        plot_history_and_save_plot_to_file(history, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    else:
        print("Loading saved model")
        testing_dataset = get_spectrogram_datasets.get_dataset("testing", 35)
        testing_dataset = testing_dataset.cache().prefetch(tf.data.AUTOTUNE)
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
                      metrics=['accuracy', tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall()])
        test_loss, test_acc, test_precision, test_recall = model.evaluate(testing_dataset)
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print(f"TEST LOSS: {test_loss}, TEST ACC: {test_acc}, TEST PRECISION: {test_precision}, TEST RECALL: {test_recall}")
        print(f"F1 score: {f1_score}")



if __name__ == '__main__':
    main()
