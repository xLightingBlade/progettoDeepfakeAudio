from tensorflow import keras


# Altro modo di creare modelli con keras, questa è la Functional API
# Reference per l'architettura ResNet : https://arxiv.org/pdf/1603.05027v2 figura 4e (full pre-activation)
class ResNetModel:
    def __init__(self, input_shape, learning_rate):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.input = keras.layers.Input(shape=input_shape)

    def residual_block(self, model_input, filters):
        bn1 = keras.layers.BatchNormalization()(model_input)
        act1 = keras.layers.Activation('relu')(bn1)
        conv1 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first',
                                    strides=(2, 2), padding='same')(act1)
        bn2 = keras.layers.BatchNormalization()(conv1)
        act2 = keras.layers.Activation('relu')(bn2)
        conv2 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first',
                                    strides=(1, 1), padding='same')(act2)
        residual = keras.layers.Conv2D(1, (1, 1), strides=(1, 1), data_format='channels_first')(conv2)

        x = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first',
                                strides=(2, 2), padding='same', )(model_input)
        out = keras.layers.Add()([x, residual])

        return out

    def build_res_network(self):
        res1 = self.residual_block(self.input, 64)
        res2 = self.residual_block(res1, 128)
        res3 = self.residual_block(res2, 256)
        res4 = self.residual_block(res3, 512)

        act1 = keras.layers.Activation('relu')(res4)
        flatten1 = keras.layers.Flatten()(act1)
        dense1 = keras.layers.Dense(512)(flatten1)
        act2 = keras.layers.Activation('relu')(dense1)
        dense2 = keras.layers.Dense(1)(act2)
        output1 = keras.layers.Activation('sigmoid')(dense2)

        model = keras.Model(inputs=self.input, outputs=output1)

        # Compiling the model
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        model.summary()
        return model
