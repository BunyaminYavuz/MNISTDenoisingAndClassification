import tensorflow as tf

# Create a model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, filename, epochs=7):
    model = create_model()
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    model.save(filename)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    return test_acc

