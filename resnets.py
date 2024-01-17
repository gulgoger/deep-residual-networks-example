import tensorflow
from tensorflow.keras import layers, Model

def residual_block(x,filters, kernel_size =3, strides=1):
    shortcut = x
    
    # first convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # second convolutional layer
    x = layers.Conv2D(filters, kernel_size, padding = "same")(x)
    x = layers.BatchNormalization()(x)
    
    # adding the shortcut connection
    x = layers.Add()([x,shortcut])
    x = layers.Activation("relu")(x)
    return x

# Building the ResNet model
input_shape = (32,32,2)
inputs = layers.Input(shape=input_shape)
    
x = layers.Conv2D(64,7, strides=2, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

# Adding residual blocks
for _ in range(3):
    x = residual_block(x, filters=64)

# Flatten and add Dense layers for classification
num_classes = 10

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

# creating the model
model = Model(inputs, outputs)

# compiling the model
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

# summary of the model architecture
model.summary()




















































