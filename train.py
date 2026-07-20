from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_mnist
from model import build_cnn

def train_model():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_cnn()
    model.summary()

    callbacks =[
        ModelCheckpoint('../models/best_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\n Accuracy sur le test: {test_acc:.4f}")

    return history, model

if __name__ == "__main__":
    train_model()