# src/train_cnn.py
import time
import tensorflow as tf
from keras import layers, models, callbacks

from utils import load_and_prepare_for_cnn, plot_history, save_results


def build_cnn(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_and_prepare_for_cnn()


    model = build_cnn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]


    start = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1, callbacks=cb)
    train_time = time.time() - start


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


    model.save('outputs/models/cnn_model.h5')
    plot_history(history, 'cnn')
    save_results('CNN', train_time, test_loss, test_acc)


    print(f'Finished CNN: train_time={train_time:.2f}s, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')