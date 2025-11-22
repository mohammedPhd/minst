# src/train_lstm.py
import time
import tensorflow as tf
from keras import layers, models, callbacks
from utils import load_and_prepare_for_lstm, plot_history, save_results


def build_lstm(input_shape=(28,28), num_classes=10):
    model = models.Sequential([
    layers.Masking(mask_value=0., input_shape=input_shape),
    layers.LSTM(128, return_sequences=False),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_and_prepare_for_lstm()


    model = build_lstm()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]


    start = time.time()
    history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1, callbacks=cb)
    train_time = time.time() - start


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


    model.save('outputs/models/lstm_model.h5')
    plot_history(history, 'lstm')
    save_results('LSTM', train_time, test_loss, test_acc)


    print(f'Finished LSTM: train_time={train_time:.2f}s, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')