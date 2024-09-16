from tensorflow import keras
from keras import layers
from keras.regularizers import L1L2

model_width = {
    "narrow": [8, 16, 32],
    "standard": [16, 32, 64],
    "wide": [32, 64, 128]
}

def conv3x3(x, out_planes, stride=1, name=None, l1=0.0, l2=0.0):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, name=name, kernel_regularizer=L1L2(l1=l1, l2=l2) if (l1 != 0.0 or l2 != 0.0) else None)(x)

def basic_block(x, planes, stride=1, name=None, l1=0.0, l2=0.0):

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1', l1=l1, l2=l2)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2', l1=l1, l2=l2)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_f(level, input_shape, width="standard", l1=0.0, l2=0.0):
    if width not in ["narrow", "standard", "wide"]:
        raise ValueError(f'Width {width} is not supported. Must be one of "narrow", "standard", "wide".')
    widths = model_width[width]
    # level can be 3, 4, 5, 6, 7, 8, 9
    if level < 3 or level > 9:
        raise NotImplementedError(f'Level {level} is not supported.')
    xin = layers.Input(input_shape)
    x = layers.ZeroPadding2D(padding=1, name='conv1_pad')(xin)
    x = layers.Conv2D(filters=widths[0], kernel_size=3, strides=1, use_bias=False, name='conv1',  kernel_regularizer=L1L2(l1=l1, l2=l2) if (l1 != 0.0 or l2 != 0.0) else None)(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)

    x = basic_block(x, widths[0], name='layer1.0', l1=l1, l2=l2)
    x = basic_block(x, widths[0], name='layer1.1', l1=l1, l2=l2)
    x = basic_block(x, widths[0], name='layer1.2', l1=l1, l2=l2)

    if level >= 4:
        x = basic_block(x, widths[1], stride=2, name='layer2.0', l1=l1, l2=l2)
    if level >= 5:
        x = basic_block(x, widths[1], name='layer2.1', l1=l1, l2=l2)
    if level >= 6:
        x = basic_block(x, widths[1], name='layer2.2', l1=l1, l2=l2)
    if level >= 7:
        x = basic_block(x, widths[2], stride=2, name='layer3.0', l1=l1, l2=l2)
    if level >= 8:
        x = basic_block(x, widths[2], name='layer3.1', l1=l1, l2=l2)
    if level >= 9:
        x = basic_block(x, widths[2], name='layer3.2', l1=l1, l2=l2)
    return keras.Model(xin, x)

# note that include_h does not mean that g includes h, it means u shape, where we remove h from g and include h separately
def make_g(level, input_shape, num_classes=10, include_h=False, dropout=0.0, width="standard", l1=0.0, l2=0.0):
    if width not in ["narrow", "standard", "wide"]: 
        raise ValueError(f'Width {width} is not supported. Must be one of "narrow", "standard", "wide".')
    if level < 3 or level > 9:
        raise NotImplementedError(f'Level {level} is not supported.')
    if level == 9 and include_h:
        raise ValueError('Level 9 does not support include_h=True, server has no model.')

    widths = model_width[width]

    xin = layers.Input(input_shape)
    x = xin

    if level <= 3:
        x = basic_block(x, widths[1], stride=2, name='layer2.0', l1=l1, l2=l2)
    if level <= 4:
        x = basic_block(x, widths[1], name='layer2.1', l1=l1, l2=l2)
    if level <= 5:
        x = basic_block(x, widths[1], name='layer2.2', l1=l1, l2=l2)
    if level <= 6:
        x = basic_block(x, widths[2], stride=2, name='layer3.0', l1=l1, l2=l2)
    if level <= 7:
        x = basic_block(x, widths[2], name='layer3.1', l1=l1, l2=l2)
    if level <= 8:
        x = basic_block(x, widths[2], name='layer3.2', l1=l1, l2=l2)
    
    if not include_h: # no h, last layers belong to g
        x = layers.GlobalAveragePooling2D(name='avgpool')(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(units=num_classes, name='fc', kernel_regularizer=L1L2(l1,l2) if (l1 != 0.0 or l2 != 0.0) else None)(x)
        return keras.Model(xin, x)
    if include_h:
        # last layers belong to h
        return keras.Model(xin, x)

def make_h(num_classes=10, dropout=0.0, l1=0.0, l2=0.0):
    model = keras.Sequential()
    model.add(layers.GlobalAveragePooling2D(name='avgpool'))
    if dropout > 0.0:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(units=num_classes, name='fc', kernel_regularizer=L1L2(l1,l2) if (l1 != 0.0 or l2 != 0.0) else None))
    return model