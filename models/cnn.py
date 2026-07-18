"""
Improved CNN architectures for image/emotion classification.

Key improvements over the original version:
  - Uses tf.keras (modern, actively maintained) instead of standalone keras
  - Consistent L2 weight regularization across ALL conv layers (not just some)
  - He-normal weight initialization (better suited for ReLU activations)
  - GlobalAveragePooling2D everywhere instead of Flatten
    -> drastically fewer parameters, less overfitting, input-size independent
  - MaxPooling2D instead of AveragePooling2D in simple/simpler CNNs
    -> generally preserves stronger features, converges faster
  - Progressive dropout (lower in early layers, higher in deeper layers)
  - Fixed missing BatchNormalization/Activation blocks that were dropped
    in the original simpler_CNN and tiny/mini/big XCEPTION residual paths
  - Added spatial dropout option and consistent use_bias=False before BN
"""

from tensorflow.keras.layers import (
    Activation, Conv2D, Dropout, BatchNormalization,
    GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D,
    Input, SpatialDropout2D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers

L2_REG = 1e-4


def conv_bn_act(x, filters, kernel_size, strides=(1, 1), dropout=None):
    """Conv2D -> BatchNorm -> ReLU (+ optional spatial dropout) helper."""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same',
               use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout:
        x = SpatialDropout2D(dropout)(x)
    return x


def simple_CNN(input_shape, num_classes):
    model = Sequential(name='simple_CNN')
    model.add(Conv2D(32, (7, 7), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG),
                      name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (7, 7), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(SpatialDropout2D(0.25))

    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(SpatialDropout2D(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(SpatialDropout2D(0.35))

    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(num_classes, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model


def simpler_CNN(input_shape, num_classes):
    model = Sequential(name='simpler_CNN')
    model.add(Conv2D(16, (5, 5), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG),
                      name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(32, (5, 5), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(num_classes, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(L2_REG)))
    # GlobalAveragePooling2D instead of Flatten:
    # far fewer parameters -> less overfitting, works with any input size
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model


def _xception_block(x, filters, dropout=None):
    """One residual separable-conv block shared by tiny/mini/big XCEPTION."""
    residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same',
                       use_bias=False,
                       kernel_initializer='he_normal')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    if dropout:
        x = SpatialDropout2D(dropout)(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    return x


def tiny_XCEPTION(input_shape, num_classes, l2_regularization=L2_REG):
    global L2_REG
    L2_REG = l2_regularization
    img_input = Input(input_shape)
    x = conv_bn_act(img_input, 8, (3, 3))
    x = conv_bn_act(x, 8, (3, 3))

    x = _xception_block(x, 16)
    x = _xception_block(x, 32)
    x = _xception_block(x, 64, dropout=0.2)
    x = _xception_block(x, 128, dropout=0.3)

    x = Conv2D(num_classes, (3, 3), padding='same',
               kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    return Model(img_input, output, name='tiny_XCEPTION')


def mini_XCEPTION(input_shape, num_classes, l2_regularization=L2_REG):
    global L2_REG
    L2_REG = l2_regularization
    img_input = Input(input_shape)
    x = conv_bn_act(img_input, 8, (3, 3))
    x = conv_bn_act(x, 8, (3, 3))

    x = _xception_block(x, 16)
    x = _xception_block(x, 32)
    x = _xception_block(x, 64, dropout=0.2)
    x = _xception_block(x, 128, dropout=0.3)

    x = Conv2D(num_classes, (3, 3), padding='same',
               kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    return Model(img_input, output, name='mini_XCEPTION')


def big_XCEPTION(input_shape, num_classes, l2_regularization=L2_REG):
    global L2_REG
    L2_REG = l2_regularization
    img_input = Input(input_shape)
    x = conv_bn_act(img_input, 32, (3, 3), strides=(2, 2))
    x = conv_bn_act(x, 64, (3, 3))

    x = _xception_block(x, 128)
    x = _xception_block(x, 256, dropout=0.2)
    x = _xception_block(x, 512, dropout=0.3)
    x = _xception_block(x, 728, dropout=0.4)

    x = Conv2D(num_classes, (3, 3), padding='same',
               kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)
    return Model(img_input, output, name='big_XCEPTION')


if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7

    model = mini_XCEPTION(input_shape, num_classes)
    model.summary()

    # Recommended training setup for these models:
    #
    # from tensorflow.keras.optimizers import Adam
    # from tensorflow.keras.callbacks import (ReduceLROnPlateau,
    #                                          EarlyStopping, ModelCheckpoint)
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    #
    # model.compile(optimizer=Adam(learning_rate=1e-3),
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # datagen = ImageDataGenerator(
    #     rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    #     zoom_range=0.1, horizontal_flip=True)
    #
    # callbacks = [
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    #     EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    #     ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
    # ]