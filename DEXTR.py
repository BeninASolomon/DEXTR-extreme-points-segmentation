from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Layer
import os 
import cv2
dataset_path = 'rough\\'

# Load images and labels
images_ct = []
labels = []
Fea1=[]
for r, d, f in os.walk(dataset_path):
    for file in f:
        images_ct.append(os.path.join(r, file))
        

# Load and preprocess images


# Process each image
for i in range(0,len(images_ct)):
    # Read the high-resolution image
    img = cv2.imread(images_ct[i])
    # Resize to high-resolution target size (example 100x100)
    fea = cv2.resize(img, (100, 100))
    gray_fea= cv2.cvtColor(fea, cv2.COLOR_BGR2GRAY)
    Fea1.append(gray_fea)
# Define the DEXTR model
Fea1 = np.array(Fea1, dtype='float32') / 255
class ExtremePointsLayer(Layer):
    def __init__(self, num_extreme_points, **kwargs):
        super(ExtremePointsLayer, self).__init__(**kwargs)
        self.num_extreme_points = num_extreme_points

    def build(self, input_shape):
        super(ExtremePointsLayer, self).build(input_shape)

    def call(self, inputs):
        extreme_points, feature_map = inputs
        reshaped_points = tf.reshape(extreme_points, (-1, self.num_extreme_points * 2))
        reshaped_points = tf.expand_dims(tf.expand_dims(reshaped_points, axis=1), axis=1)
        tiled_points = tf.tile(reshaped_points, [1, tf.shape(feature_map)[1], tf.shape(feature_map)[2], 1])
        return tiled_points

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2], self.num_extreme_points * 2)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

def dextr_net(input_shape=(100, 100, 1), num_extreme_points=4):
    # Input for image data
    inputs = Input(input_shape)

    # Additional input for extreme points
    extreme_points = Input(shape=(num_extreme_points, 2))  # Assuming 2D coordinates for each extreme point

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up3 = UpSampling2D(size=(2, 2))(conv3)
    up3 = Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge3 = concatenate([conv2, up3], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = Conv2D(64, 2, activation='relu', padding='same')(up2)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Use the custom ExtremePointsLayer to reshape and tile the extreme points
    tiled_extreme_points = ExtremePointsLayer(num_extreme_points)([extreme_points, conv5])

    # Concatenate extreme points with conv5
    concat_extreme_points = concatenate([conv5, tiled_extreme_points], axis=-1)

    # Final convolutional layer for segmentation output
    outputs = Conv2D(1, 1, activation='sigmoid')(concat_extreme_points)  # Binary segmentation (1 channel)

    # Define model with inputs and outputs
    model = Model(inputs=[inputs, extreme_points], outputs=outputs)
    return model

# Example usage
input_shape = (100, 100, 1)  # Adjust based on your image size and channels
num_extreme_points = 4  # Number of extreme points used in DEXTR

# Create and compile the DEXTR model
model = dextr_net(input_shape=input_shape, num_extreme_points=num_extreme_points)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Example extreme points (replace with your actual data loading and preprocessing)
E = np.random.rand(len(Fea1), num_extreme_points, 2)  # Example extreme points

# # Train the DEXTR model
# history = model.fit([Fea1, E], Fea2, epochs=500, batch_size=5)

# # # Save the model
# model.save('dextr_model.h5')

model = keras.models.load_model('dextr_model.h5', custom_objects={
  'ExtremePointsLayer': ExtremePointsLayer})
# Use the model for prediction
hh = model.predict([Fea1, E])

# Plot predictions for the first image
# plt.imshow(Fea1[1])
# plt.show()
for h in range(0,len(hh)):
    plt.imshow(hh[h])
    plt.show()

