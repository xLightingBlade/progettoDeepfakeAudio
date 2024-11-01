from keras_tuner import HyperModel
import tensorflow as tf

class MyCnnHyperModel(HyperModel):
    def __init__(self, input_shape, number_of_layers, dropout_rate, learning_rate):
        self.input_shape = input_shape
        self.number_of_layers = number_of_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self, hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=self.input_shape))
        # reg. l2 assieme al dropout: studi mostrano come porta maggior riduzione della loss
        model.add(tf.keras.layers.Conv2D(filters=hp.Int('first_conv_filters', min_value=16, max_value=64, step=16),
                                         kernel_size=hp.Choice('first_conv_kernel', values=[2, 5]), activation='relu',
                                         padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

        if self.number_of_layers > 1:
            model.add(tf.keras.layers.Conv2D(filters=hp.Int('second_conv_filters', min_value=16, max_value=64, step=16),
                                             kernel_size=hp.Choice('second_conv_kernel', values=[2, 5]),
                                             activation='relu',
                                             padding='same',
                                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            if self.number_of_layers > 2:
                model.add(
                    tf.keras.layers.Conv2D(filters=hp.Int('third_conv_filters', min_value=16, max_value=64, step=16),
                                           kernel_size=hp.Choice('third_conv_kernel', values=[2, 5]),
                                           activation='relu',
                                           padding='same',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
            if self.number_of_layers > 3:
                model.add(
                    tf.keras.layers.Conv2D(filters=hp.Int('fourth_conv_filters', min_value=16, max_value=64, step=16),
                                           kernel_size=hp.Choice('fourth_conv_kernel', values=[2, 5]),
                                           activation='relu',
                                           padding='same',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=64,
                                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)))

        model.add(tf.keras.layers.Dropout(self.dropout_rate))

        # output layer, funzione di attivazione sigmoide per class. binaria, 1 neurone per cardinalità output,
        # quindi uno
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="binary_crossentropy",
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        model.summary()

        return model
