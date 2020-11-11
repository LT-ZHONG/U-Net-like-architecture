import numpy as np
import tensorflow as tf
from PIL import ImageOps
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, array_to_img

from model import get_model
from dataPreprocess import prepare_path, make_validation, Oxford

input_dir = '/home/hello/PycharmProjects/UNet/images'
target_dir = '/home/hello/PycharmProjects/UNet/annotations/trimaps'

image_size = (160, 160)
num_classes = 4
batch_size = 32

input_image_paths, target_image_paths = prepare_path(input_dir=input_dir, target_dir=target_dir)

# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

# Split our img paths into a training and a validation set
train_input_image_paths, train_target_image_paths, validate_input_image_paths, validate_target_image_paths = \
    make_validation(input_image_paths=input_image_paths, target_image_paths=target_image_paths)

# Instantiate data Sequences for each split
train_generator = Oxford(
    bat_size=batch_size,
    img_size=image_size,
    input_img_path=train_input_image_paths,
    target_img_path=train_target_image_paths
)

validate_generator = Oxford(
    bat_size=batch_size,
    img_size=image_size,
    input_img_path=validate_input_image_paths,
    target_img_path=validate_target_image_paths
)


''' Build model '''
model = get_model(img_size=image_size, num_classes=num_classes)

model.load_weights(filepath='oxford_segmentation.h5')

# model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]

epochs = 50
model.fit(train_generator, epochs=epochs, validation_data=validate_generator, callbacks=callbacks)  # starting training

model.load_weights(filepath='oxford_segmentation.h5')


''' Visualize predictions '''
# Generate predictions for all images in the validation set
testing_generator = Oxford(bat_size=batch_size,
                           img_size=image_size,
                           input_img_path=validate_input_image_paths,
                           target_img_path=validate_target_image_paths)

testing_predictions = model.predict(testing_generator)
print(testing_predictions.shape)

# Display results for validation image #100
i = 100

# Display input image
img = load_img(path=validate_input_image_paths[i])
plt.imshow(img)
plt.show()

# Display ground-truth target mask
img = ImageOps.autocontrast(load_img(validate_target_image_paths[i]))
plt.imshow(img)
plt.show()


def display_mask(j):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(testing_predictions[j], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img_prediction = ImageOps.autocontrast(image=array_to_img(mask))

    plt.imshow(img_prediction)
    plt.show()


# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.
