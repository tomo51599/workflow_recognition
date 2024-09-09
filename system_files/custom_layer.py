import einops
import keras
from keras.layers import Layer
from keras.utils import register_keras_serializable
from keras import layers

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()


@keras.saving.register_keras_serializable(package="MyCustomLayers")
class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super().__init__(**kwargs)  # 親クラスのコンストラクタを呼び出す
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seq = keras.Sequential([
            # Spatial decomposition
            keras.layers.Conv3D(filters=self.filters,
                                kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                                padding=self.padding),
            # Temporal decomposition
            keras.layers.Conv3D(filters=self.filters,
                                kernel_size=(self.kernel_size[0], 1, 1),
                                padding=self.padding)
        ])
  
    def call(self, x):
        return self.seq(x)
    
    def get_config(self):
        config = super().get_config()  # 親クラスの設定を取得
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding
        })
        return config

@keras.saving.register_keras_serializable(package="CustomLayers")
class ResidualMain(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualMain, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])
  
    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super(ResidualMain, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

@keras.saving.register_keras_serializable(package="CustomLayers")
class Project(Layer):
    def __init__(self, units, **kwargs):
        super(Project, self).__init__(**kwargs)
        self.units = units
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super(Project, self).get_config()
        config.update({'units': self.units})
        return config

@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def add_residual_block(input, filters, kernel_size):

  out = ResidualMain(filters, 
                     kernel_size)(input)
  
  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

@keras.saving.register_keras_serializable(package="CustomLayers")
class ResizeVideo(Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeVideo, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        # b stands for batch size, t stands for time, h stands for height, 
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos

    def get_config(self):
        config = super(ResizeVideo, self).get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config
  
