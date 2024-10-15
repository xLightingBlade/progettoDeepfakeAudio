from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM


def build_rnn_model(model_type, input_shape, learning_rate):
    model = keras.models.Sequential()
    if model_type == 'lstm_simple':
        model.add(LSTM(units=128,return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
    else:
        raise Exception('Wrong model type')
    model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
