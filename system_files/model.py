from keras import layers, models
from system_files.custom_layer import Conv2Plus1D, ResidualMain, Project, ResizeVideo, add_residual_block

HEIGHT = 224
WIDTH = 224

input_shape = (None, 48, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(6)(x)
#layers.Dense(6, activation='softmax')(x)

custom_objects = {
    "Conv2Plus1D": Conv2Plus1D,
    "ResidualMain": ResidualMain,
    "Project": Project,
    "ResizeVideo": ResizeVideo,
    "add_residual_block": add_residual_block  
}