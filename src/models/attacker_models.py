from tensorflow import keras
from keras import layers

def make_decoder(level, in_shape, conditional, num_class, deep=True, width="standard"):
    model_width = {
        "narrow": [8, 16, 32],
        "standard": [16, 32, 64],
        "wide": [32, 64, 128]
    }
    if width not in {"narrow", "standard", "wide"}:
        raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")
    widths = model_width[width]
    if conditional:
        yin = layers.Input(shape=(1,)) # label input Layer
        xin = layers.Input(in_shape)
        yin_embedding = layers.Embedding(num_class, 50)(yin) # Embed label to vector
        n_nodes = in_shape[0] * in_shape[1]
        yin_embedding = layers.Dense(n_nodes)(yin_embedding)
        yin_embedding = layers.Reshape((in_shape[0], in_shape[1], 1))(yin_embedding) # New shape

        x = layers.Concatenate()([xin, yin_embedding]) # Concatenate
    else:
        xin = layers.Input(in_shape)
        x = xin
    if deep and level >= 9:
        x = layers.Conv2DTranspose(widths[2], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if deep and level >= 8:
        x = layers.Conv2DTranspose(widths[2], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if level >= 7:
        # x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
        # replace conv2dtranspose with upsample + conv2d
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(widths[2], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if deep and level >= 6:
        x = layers.Conv2DTranspose(widths[1], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if deep and level >= 5:
        x = layers.Conv2DTranspose(widths[1], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if level >= 4:
        # x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)
        # replace conv2dtranspose with upsample + conv2d
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Conv2D(widths[1], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    if deep and level >= 3:
        x = layers.Conv2DTranspose(widths[0], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(widths[0], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(widths[0], (3, 3), strides=(1, 1), padding="same")(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.ReLU()(x)
    x = layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation="sigmoid")(x)
    if conditional:
        model = keras.models.Model([xin, yin], x)
    else:
        model = keras.models.Model(xin, x)
    return model

def make_simulator_discriminator(level, in_shape, conditional, num_class, width="standard", bn=True):
    if width not in {"narrow", "standard", "wide"}:
        raise ValueError("width must be one of {'narrow', 'standard', 'wide'}")
    if width == "standard":
        widths = [32, 64, 128, 256]
    elif width == "narrow":
        widths = [16, 32, 64, 128]
    elif width == "wide":
        widths = [64, 128, 256, 512]

    if conditional:
        xin = layers.Input(in_shape)
        yin = layers.Input(shape=(1,)) # label input Layer
        yin_embedding = layers.Embedding(num_class, 50)(yin) # Embed label to vector
        n_nodes = in_shape[0] * in_shape[1]
        yin_embedding = layers.Dense(n_nodes)(yin_embedding)
        yin_embedding = layers.Reshape((in_shape[0], in_shape[1], 1))(yin_embedding) # New shape

        x = layers.Concatenate()([xin, yin_embedding]) # Concatenate
    else:
        xin = layers.Input(in_shape)
        x = xin

    if level == 3: # input_shape = (32,32,16)
        x = layers.Conv2D(widths[0], (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[1], (3, 3), strides=(2, 2), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[2], (3, 3), strides=(2, 2), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(2, 2), padding='same')(x)
    elif level <= 6:
        x = layers.Conv2D(widths[1], (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[2], (3, 3), strides=(2, 2), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)   
        x = layers.Conv2D(widths[3], (3, 3), strides=(2, 2), padding='same')(x)
    elif level <= 9:
        x = layers.Conv2D(widths[2], (3, 3), strides=(1, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(1, 1), padding='same')(x)
        if bn:
            x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(widths[3], (3, 3), strides=(2, 2), padding='same')(x)
    # classifier
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1)(x)

    if conditional:
        model = keras.models.Model([xin, yin], x)
    else:
        model = keras.models.Model(xin, x)
    return model

def make_decoder_discriminator(in_shape, conditional, num_class):
    if conditional:
        xin = layers.Input(in_shape)
        yin = layers.Input(shape=(1,)) # label input Layer
        yin_embedding = layers.Embedding(num_class, 50)(yin) # Embed label to vector
        n_nodes = in_shape[0] * in_shape[1]
        yin_embedding = layers.Dense(n_nodes)(yin_embedding)
        yin_embedding = layers.Reshape((in_shape[0], in_shape[1], 1))(yin_embedding) # New shape

        x = layers.Concatenate()([xin, yin_embedding]) # Concatenate
    else:
        xin = layers.Input(in_shape)
        x = xin
    
    # normal
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # downsample
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    #downsample
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    #downsample
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # classifier
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(1)(x)
    if conditional:
        model = keras.models.Model([xin, yin], x)
    else:
        model = keras.models.Model(xin, x)
    return model