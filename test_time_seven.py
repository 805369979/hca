import csv

import cv2
import keras_cv_attention_models
from sklearn.metrics import f1_score, recall_score
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers
import os
import numpy as np
import warnings
import time
import tensorflow.keras.layers
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
import sklearn
import seaborn as sns
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add, \
    ReLU, MaxPool2D, Reshape, GlobalAvgPool2D, Multiply
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras import layers as L

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
# import keras_cv_attention_models.fastervit
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
# -------------------------------------------------
# 超参数
# -------------------------------------------------

import cbam
import numpy as np
import uuid
# 用于避免卷积层同名报错
unique_random_number = uuid.uuid4()
# import keras_cv_attention_models.fastervit
from sklearn.model_selection import train_test_split

# -------------------------------------------------
#  LW-Former = EfficientNetB0-Shallow + CrossFormer
# -------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow.keras.applications import EfficientNetB0


def swin_mlp_block(x, dim, window_size=7, mlp_ratio=4):
    """Window-MLP block: LayerNorm → Window partition → MLP → reverse window → Add"""
    B, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-5)(x)

    # Pad to multiple of window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = layers.ZeroPadding2D(((0, pad_h), (0, pad_w)))(x)
    _, Hp, Wp, _ = x.shape

    # Reshape into windows (B, Nw, window_size, window_size, C)
    x = layers.Reshape((Hp // window_size, window_size,
                        Wp // window_size, window_size, C))(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])  # (B, H/w, W/w, w, w, C)
    x = layers.Reshape((-1, window_size * window_size, C))(x)  # (B*Nw, w*w, C)

    # MLP along spatial dimension
    x = layers.Dense(window_size * window_size, activation='gelu')(x)
    x = layers.Dense(C)(x)

    # Restore spatial shape
    x = layers.Reshape((Hp // window_size, Wp // window_size,
                        window_size, window_size, C))(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = layers.Reshape((Hp, Wp, C))(x)
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = layers.Cropping2D(((0, pad_h), (0, pad_w)))(x)

    # Channel MLP
    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Dense(int(dim * mlp_ratio), activation='gelu')(x)
    x = layers.Dense(dim)(x)
    return layers.Add()([shortcut, x])
# 1. 超参数
IMG_SIZE   = 224          # CIFAR-10 可设 32；若 224 会自动 resize
BATCH_SIZE = 128
EPOCHS     = 200
NUM_CLASS  = 10
MIXUP_ALPHA= 0.2
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def mobilenet_block(x, filters, kernel_size, strides=1):
    """
    MobileNet Block: Depthwise Convolution + Pointwise Convolution
    """
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def convnext_block(x, filters):
    """
    ConvNeXt Block: Depthwise Convolution + LayerNorm + MLP
    """
    # Ensure the number of groups matches the number of input channels
    input_channels = x.shape[-1]
    x = layers.Conv2D(input_channels, kernel_size=7, padding='same', groups=input_channels)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(4 * filters, activation='gelu')(x)
    x = layers.Dense(filters)(x)
    return x

def efficientnet_block(x, filters, kernel_size, strides=1, expand_ratio=6):
    """
    EfficientNet Block: MBConv Block with Swish activation
    """
    input_channels = x.shape[-1]
    expanded_channels = input_channels * expand_ratio
    input = x
    if expand_ratio > 1:
        x = layers.Conv2D(expanded_channels, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides == 1 and input_channels == filters:
        x = layers.Add()([x, layers.Conv2D(filters, 1, padding='same', use_bias=False)(input)])
    return x

IMAGE_ORDERING = 'channels_last'
class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 92
        self.batchsize = 16
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.history = self.train()
        end = time.time()
        print(f'训练共耗时{round(end - start, 2)}s')

        # self.show_history()

    @staticmethod
    def get_call_backs():
        call_backs = [
            # ModelCheckpoint('./logs/' + 'best.h5',
            #                 save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
            # EarlyStopping(monitor='val_loss', patience=40)
        ]

        print("调用回调函数!!!")

        return call_backs



    def SKBlock(self,input_tensor, filters):
        # 分支1: 3x3卷积
        # branch1 = SeparableConv2D(filters, 3, padding='same', activation='relu')(input_tensor)

        branch2 = SeparableConv2D(filters, 5, padding='same', activation='relu')(input_tensor)
        # 分支2: 5x5卷积（等效两个3x3）
        branch3 = SeparableConv2D(filters, 7, padding='same', activation='relu',name="gram-image")(input_tensor)

        # 注意力融合
        merged = Concatenate()([branch2,branch3])
        gap = GlobalAvgPool2D()(merged)
        fc = Dense(filters // 8, activation='relu')(gap)
        attn = Dense(filters * 2, activation='softmax')(fc)  # 动态权重
        attn_b2, attn_b3 = tf.split(attn, num_or_size_splits=2, axis=1)
        output =  Multiply()([branch2, attn_b2])+Multiply()([branch3, attn_b3])

        # 残差连接
        shortcut = Conv2D(filters, 1)(input_tensor) if input_tensor.shape[-1] != filters else input_tensor
        return tf.keras.layers.Add()([output, shortcut])

    def sobel_branch(self, inp, view_idx=0):
        """Sobel 边缘检测分支（每个视角独立，名称加 view_idx 后缀）"""
        p = f'v{view_idx}_'
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_x = np.stack([sobel_x] * 3, axis=-1)[..., None]
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        kernel_y = np.stack([sobel_y] * 3, axis=-1)[..., None]

        edge = layers.Conv2D(1, 3, padding='same', use_bias=False,
                             weights=[kernel_x], trainable=False, name=f'{p}sobel_x')(inp)
        edgey = layers.Conv2D(1, 3, padding='same', use_bias=False,
                             weights=[kernel_y], trainable=False, name=f'{p}sobel_y')(inp)
        edge1 = layers.add([edge, edgey])
        edge1 = layers.MaxPooling2D(name=f'{p}s_mp0')(edge1)

        edge = SeparableConv2D(64, (9, 9), activation='relu', padding='same', name=f'{p}s_sep1')(edge1)
        edge = layers.MaxPooling2D(name=f'{p}s_mp1')(edge)
        edge1 = SeparableConv2D(64, (3, 3), activation='relu', padding='same', name=f'{p}s_sep2')(edge)
        edge1 = SeparableConv2D(64, (5, 5), activation='relu', padding='same', name=f'{p}s_sep3')(edge1)
        edge1 = SeparableConv2D(64, (7, 7), activation='relu', padding='same', name=f'{p}s_sep4')(edge1)
        edge = add([edge1, edge], name=f'{p}s_add1')
        edge = SeparableConv2D(128, (7, 7), activation='relu', padding='same', name=f'{p}s_sep5')(edge)
        edge = layers.MaxPooling2D(name=f'{p}s_mp2')(edge)
        edge1 = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name=f'{p}s_sep6')(edge)
        edge1 = SeparableConv2D(128, (5, 5), activation='relu', padding='same', name=f'{p}s_sep7')(edge1)
        edge1 = SeparableConv2D(128, (7, 7), activation='relu', padding='same', name=f'{p}s_sep8')(edge1)
        edge = add([edge1, edge], name=f'{p}s_add2')
        edge = SeparableConv2D(256, (5, 5), activation='relu', padding='same', name=f'{p}s_sep9')(edge)
        edge = layers.MaxPooling2D(name=f'{p}s_mp3')(edge)
        edge1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name=f'{p}s_sep10')(edge)
        edge1 = SeparableConv2D(256, (5, 5), activation='relu', padding='same', name=f'{p}s_sep11')(edge1)
        edge1 = SeparableConv2D(256, (7, 7), activation='relu', padding='same', name=f'{p}s_sep12')(edge1)
        edge = add([edge1, edge], name=f'{p}s_add3')
        edge = SeparableConv2D(1024, (3, 3), activation='relu', padding='same', name=f'{p}s_sep13')(edge)
        edge = layers.MaxPooling2D(name=f'{p}s_mp4')(edge)
        edge = cbam.cbam_module(edge)
        return edge

    def build_model(self):
        inputs = Input(shape=self.x_shape)

        def sobel_branch(inp):
            # Sobel x
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            kernel = np.stack([sobel_x] * 3, axis=-1)[..., None]  # (3,3,3,1)
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            kernely = np.stack([sobel_y] * 3, axis=-1)[..., None]  # (3,3,3,1)

            edge = layers.Conv2D(1, 3, padding='same', use_bias=False,
                                 weights=[kernel], trainable=False, name='sobel')(inp)
            edgey = layers.Conv2D(1, 3, padding='same', use_bias=False,
                                 weights=[kernely], trainable=False, name='sobel1')(inp)
            edge1 = layers.add([edge, edgey])  # 双通道融合

            edge = SeparableConv2D(64, (9, 9), activation='relu', padding='same')(edge1)
            edge = layers.MaxPooling2D()(edge)

            edge1 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(edge)
            edge1 = SeparableConv2D(64, (5, 5), activation='relu', padding='same')(edge1)
            edge1 = SeparableConv2D(64, (7, 7), activation='relu', padding='same')(edge1)

            edge = add([edge1,edge])

            edge = SeparableConv2D(128, (7, 7), activation='relu', padding='same')(edge)
            edge = layers.MaxPooling2D()(edge)

            edge1 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(edge)
            edge1 = SeparableConv2D(128, (5, 5), activation='relu', padding='same')(edge1)
            edge1 = SeparableConv2D(128, (7, 7), activation='relu', padding='same')(edge1)
            edge = add([edge1, edge])

            edge = SeparableConv2D(256, (5, 5), activation='relu', padding='same')(edge)
            edge = layers.MaxPooling2D()(edge)

            edge1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(edge)
            edge1 = SeparableConv2D(256, (5, 5), activation='relu', padding='same')(edge1)
            edge1 = SeparableConv2D(256, (7, 7), activation='relu', padding='same')(edge1)

            edge = add([edge1, edge])

            edge = SeparableConv2D(1024, (3, 3), activation='relu', padding='same')(edge)
            edge = layers.MaxPooling2D()(edge)
            edge = cbam.cbam_module(edge)
            return edge

        soble = self.sobel_branch(inputs)

        base = tensorflow.keras.applications.MobileNet(include_top=False, weights='imagenet',input_tensor=inputs)
        # print(base.summary())
        base = base.get_layer('conv_pw_13_relu')

        soble = GlobalAveragePooling2D()(soble)
        base = GlobalAveragePooling2D()(base.output)
        x = multiply([base,soble])

        # x = concatenate([base.output,x])
        # x2 = GlobalAveragePooling2D()(x2)
        # x1 = GlobalAveragePooling2D()(x1)
        x = self.dynamic_weight_fusion(base,x)
        # x = concatenate([base,x])
        # x = Dense(128, activation='relu')(x)
        outputs = Dense(self.classes, activation="softmax",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=inputs, outputs=outputs)
        model_.summary()
        return model_

    def dynamic_weight_fusion(self, base, scale):
        # ‌通道注意力增强版（CA - EfficientNet改进方案）‌
        # 动态权重生成（双层Dense）
        weights = layers.Concatenate()([base, scale])
        # weights = BatchNormalization()(weights)
        weights = layers.Dense(64, activation='relu')(weights)
        weights = layers.Dense(2, activation='softmax')(weights)
        # 加权融合
        weighted_base = layers.Multiply()([base, weights[:, 0:1]])
        weighted_scale = layers.Multiply()([scale, weights[:, 1:2]])

        return layers.Add()([weighted_base, weighted_scale])

    def train(self):
        import numpy as np
        batch_size = 64
        x = tf.constant(np.random.randn(batch_size, 224, 224, 3))

        # 预热（不计时）
        print("预热 10 次...")
        for i in range(10):
            _ = self.model.predict(x)

        # 测 FPS（batch_size=1，反映单样本推理速度）
        num_runs = 500
        print(f"正式测量 {num_runs} 次（batch_size={batch_size}）...")
        start = time.time()
        for i in range(num_runs):
            features = self.model.predict(x)
        end = time.time() - start
        print(end/num_runs)
        total_samples = num_runs * batch_size
        avg_latency_ms = (end / num_runs) * 1000     # 每次predict平均延迟(ms)
        fps = total_samples / end                    # 每秒处理样本数
        print(f"总耗时: {end:.4f} s")
        print(f"总样本数: {total_samples}")
        print(f"平均延迟: {avg_latency_ms:.2f} ms/batch")
        print(f"FPS: {fps:.2f} samples/s")

        # 计算 GMACs
        x = tf.constant(np.random.randn(batch_size, 224, 224, 3))
        print(get_flops(self.model, [x]))



def get_flops(model, model_inputs) -> float:
    """
    Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
    in inference mode. It uses tf.compat.v1.profiler under the hood.
    """
    # if not hasattr(model, "model"):
    #     raise wandb.Error("self.model must be set before using this method.")

    if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
    ):
        raise ValueError(
            "Calculating FLOPS is only supported for "
            "`tf.keras.Model` and `tf.keras.Sequential` instances."
        )

    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
        for inp in model_inputs
    ]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
            .with_empty_output()
            .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    # convert to GFLOPs
    return (flops.total_float_ops / 1e9) / 2

if __name__ == '__main__':

    # 测试自己的模型时间
    import tensorflow as tf
    # 使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(tf.test.is_gpu_available())
    # fer_model = FerModel()
    # print(tf.test.is_gpu_available())

    # # 测试模型时间
    # import tensorflow as tf
    # # 使用gpu则开启这一行代码
    # # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # print(tf.train.is_gpu_available())
    # import numpy as np
    base_model = tensorflow.keras.applications.efficientnet.EfficientNetB7(input_shape=(224,224,3),weights=None,include_top=False)
    # base_model = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.resnet.ResNet152(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.NASNetMobile(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.Xception(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # base_model = tensorflow.keras.applications.(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(10, activation='softmax')(x)
    print(base_model.summary())
    model = Model(inputs=base_model.input, outputs=predictions)
    x = tf.constant(np.random.randn(32, 224, 224, 3))
    start = time.time()
    # cpu 50 Gpu 500
    #z
    for i in range(500):
        features = model.predict(x)
    end = time.time() - start
    print(end / i)
    x = tf.constant(np.random.randn(64, 224, 224, 3))
    print(get_flops(model, [x]))


    # ======================================================================================
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # # 使用cpu则开启这一行代码
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # base_model = keras_cv_attention_models.swin_transformer_v2.SwinTransformerV2(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fastervit.FasterViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fastvit.FastViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.mobilevit.MobileViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.uniformer.Uniformer(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.davit.DaViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fastvit.FastViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.flexivit.FlexiViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.mobilevit.MobileViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.convnext.ConvNeXtV2(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.efficientnet.EfficientNetV2(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.resnext.ResNeXt50(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fbnetv3.FBNetV3D(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fbnetv3.FBNetV3G(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.fbnetv3.FBNetV3B(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.vit.ViT(input_shape=(224, 224, 3))
    # base_model = keras_cv_attention_models.lcnet.LCNet(input_shape=(224, 224, 3))
    # base = base_model.layers[-2].output
    # predictions = Dense(10, activation="softmax", kernel_initializer='he_normal')(base)
    # model = keras.Model(inputs=base_model.inputs, outputs=predictions)
    # print(model.summary())
    # x = tf.constant(np.random.randn(64, 224, 224, 3))
    # start = time.time()
    # # Gpu 500
    # for i in range(500):
    #     features = model.predict(x)
    # end = time.time() - start
    # print(end / i)
    # x = tf.constant(np.random.randn(1, 224, 224, 3))
    # print(get_flops(model, [x]))