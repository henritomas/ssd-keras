from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = 3 #last index is channels
    filters = int(filters * alpha)

    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    return x

def _depthwise_conv_block_classification(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
 
    channel_axis = 3
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_pw_%d_relu' % block_id)(x)
    return x

def mobilenet(input_tensor, alpha=1.0, depth_multiplier=1):

    if input_tensor is None:
        input_tensor = Input(shape=(300,300,3))

    x = _conv_block(input_tensor, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block_classification(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block_classification(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block_classification(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=10)
    conv4_3 = _depthwise_conv_block_classification(x, 512, alpha, depth_multiplier, block_id=11) #11 conv4_3 (300x300)-> 19x19 

    x = _depthwise_conv_block_classification(conv4_3, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)   # (300x300) -> 10x10 
    fc7 = _depthwise_conv_block_classification(x, 1024, alpha, depth_multiplier, block_id=13) # 13 fc7 (300x300) -> 10x10

    #model = Model(inputs=input_tensor, outputs=fc7)
    #return model

    return [conv4_3, fc7]