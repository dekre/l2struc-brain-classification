import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def get_tf_cnn(
        num_classes: int = 2, 
        img_height: int = 128, 
        img_width: int = 128, 
        optimizer: str = 'adam', 
        metrics: list = ['accuracy']
    ) -> Sequential:
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./1000, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 2, padding='same', activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(32, 2, padding='same', activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Conv2D(64, 2, padding='same', activation='relu'),
        layers.MaxPooling2D(padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    (
        model
        .compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics
        )
    )

    return model