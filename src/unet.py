from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose,
                          Dropout)


def conv_block(inputs, filters, dropout_rate=0.2):
    c = Conv2D(filters, (3, 3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same')(inputs)
    c = Dropout(dropout_rate)(c)
    c = Conv2D(filters, (3, 3),
               activation='relu',
               kernel_initializer='he_normal',
               padding='same')(c)
    return c


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    filters = [16, 32, 64, 128, 256]
    p = s
    c_list = []
    for filt in filters:
        c = conv_block(p, filt, dropout_rate=0.1 if filt <= 32 else 0.2)
        c_list.append(c)
        p = MaxPooling2D((2, 2))(c)

    # Expansive path
    p = c_list[-1]
    filters = filters[:-1][::-1]
    for i, filt in enumerate(filters):
        u = Conv2DTranspose(filt, (2, 2), strides=(2, 2), padding='same')(p)
        u = concatenate([u, c_list[-(i + 2)]])
        p = conv_block(u, filt, dropout_rate=0.2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(p)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
