import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (3,))

    ''' [First half of the network: down sampling inputs] '''

    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Project residual
        residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same')(previous_block_activation)

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual

    ''' [Second half of the network: upsampling inputs] '''

    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(size=2)(x)

        # Project residual
        residual = layers.UpSampling2D(size=2)(previous_block_activation)
        residual = layers.Conv2D(filters=filters, kernel_size=1, padding='same')(residual)

        x = layers.add([x, residual])  # Add back residual

        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(filters=num_classes, kernel_size=3, activation='softmax', padding='same')(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)

    plot_model(model=model, show_shapes=True)

    return model
