import cv2

from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, PReLU, Input, \
    BatchNormalization, GlobalMaxPooling2D, SeparableConv2D, LeakyReLU, Concatenate,Lambda,Flatten,SeparableConvolution2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, Flatten, AveragePooling2D, add
from tensorflow.keras import regularizers
import time
import tensorflow.keras.layers
import sklearn
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add, \
    Reshape, Conv1D, ReLU, GlobalAvgPool2D, Multiply, GlobalMaxPool2D

import cbam
import uuid
unique_random_number = uuid.uuid4()

from tensorflow.keras import layers as L
from tensorflow.keras import backend as K

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

def gp_3x3( inputs, pre_layer):
    interval = int(pre_layer / 2)

    normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
    normalized1_2 = Lambda(lambda x: x[:, :, :, interval:],name=str(uuid.uuid4()))(inputs)

    # 第一路
    tower_1 = DepthwiseConv2D((3, 3), padding='same',
                              input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_1)

    # 第二路
    tower_2 = DepthwiseConv2D((3, 3), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_2)

    output = tensorflow.keras.layers.concatenate([tower_1, tower_2], axis=3,name=str(uuid.uuid4()))
    # output = self.channel_shuffle(output, 2)

    output = Conv2D(pre_layer // 4, 1, kernel_regularizer=regularizers.l2(0.0005), activation='relu')(
        output)
    output = BatchNormalization()(output)

    return output

def gp_5x5(inputs, pre_layer):
    interval = int(pre_layer / 4)

    normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
    normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
    normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
    normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

    # 第一路
    tower_1 = DepthwiseConv2D((5, 1), padding='same',
                              input_shape=(128,128,3), kernel_initializer='uniform',
                              kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_1)
    tower_1 = DepthwiseConv2D((1, 5), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_1)

    # 第二路
    tower_2 = DepthwiseConv2D((5, 1), padding='same',
                              input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_2)
    tower_2 = DepthwiseConv2D((1, 5), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_2)

    # 第三路
    tower_3 = DepthwiseConv2D((5, 1), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_3)
    tower_3 = DepthwiseConv2D((1, 5), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_3)

    # 第四路
    tower_4 = DepthwiseConv2D((5, 1), padding='same',
                              input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_4)
    tower_4 = DepthwiseConv2D((1, 5), padding='same',
                              input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_4)

    output = tensorflow.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
    # output = self.channel_shuffle(output, 4)

    output = Conv2D(pre_layer // 4, 1,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005), activation='relu')(
        output)
    output = BatchNormalization()(output)

    return output

def gp_7x7(inputs, pre_layer):
    interval = int(pre_layer / 4)

    normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
    normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
    normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
    normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

    # 第一路
    tower_1 = SeparableConv2D(interval // 4, (7, 1), padding='same',
                              input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_1)
    tower_1 = SeparableConv2D(interval // 4, (1, 7),kernel_initializer='uniform', padding='same',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_1)

    # 第二路
    tower_2 = SeparableConv2D(interval // 4, (7, 1),kernel_initializer='uniform', padding='same',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_2)
    tower_2 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_2)

    # 第三路
    tower_3 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_3)
    tower_3 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_3)
    # 第四路
    tower_4 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(normalized1_4)
    tower_4 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                              input_shape=(128,128,3), kernel_regularizer=regularizers.l2(0.0005),
                              activation='relu')(tower_4)

    output = tensorflow.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
    output = BatchNormalization()(output)

    return output



def gp_block( inputs, in_filters):
    # 生成分组卷积模块

    # 第一路
    tower_1 = Conv2D(in_filters // 4, (1, 1), padding='same',name=str(uuid.uuid4()),kernel_initializer='uniform',
                     input_shape=(128,128,3),
                     activation='relu')(inputs)
    tower_1 = BatchNormalization()(tower_1)

    # 第二路
    tower_2 = gp_3x3(inputs, in_filters)

    # 第三路
    tower_3 = gp_5x5(inputs, in_filters)

    # 第四路
    tower_4 = gp_7x7(inputs, in_filters)

    # 第四路
    # tower_5 = self.gp_9x9(inputs, in_filters)
    # output = [tower_1, tower_2, tower_3, tower_4]

    output = tf.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
    # fused = WeightedFusion(4)(output)
    return output
def py_block( inputs, in_filters, out_filters):
    # 生成金字塔卷积模块

    weight_decay = 0.0005

    x = Conv2D(in_filters, 1,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

    x = BatchNormalization()(x)

    x = gp_block(x, in_filters)

    x = cbam.cbam_module(x)

    x = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(weight_decay),name=str(uuid.uuid4()))(x)
    x = BatchNormalization()(x)

    inputs = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(weight_decay),name=str(uuid.uuid4()))(inputs)
    inputs = BatchNormalization()(inputs)

    x = tensorflow.keras.layers.add([inputs, x])
    x = Activation('relu')(x)

    return x
def ca_block(input_feature, ratio=16, name=""):
    channel = input_feature.shape[-1]
    h = input_feature.shape[1]
    w = input_feature.shape[2]

    x_h = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    x_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_h)
    x_w = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)

    x_cat_conv_relu = Concatenate(axis=2)([x_w, x_h])
    x_cat_conv_relu = Conv2D(channel // ratio, kernel_size=1, strides=1, use_bias=False,
                             name="ca_block_conv1_" + str(name))(x_cat_conv_relu)
    x_cat_conv_relu = BatchNormalization(name="ca_block_bn_" + str(name))(x_cat_conv_relu)
    x_cat_conv_relu = Activation('relu')(x_cat_conv_relu)

    x_cat_conv_split_h, x_cat_conv_split_w = Lambda(lambda x: tf.split(x, num_or_size_splits=[h, w], axis=2))(
        x_cat_conv_relu)
    x_cat_conv_split_h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(x_cat_conv_split_h)
    x_cat_conv_split_h = Conv2D(channel, kernel_size=1, strides=1, use_bias=False,
                                name="ca_block_conv2_" + str(name))(x_cat_conv_split_h)
    x_cat_conv_split_h = Activation('sigmoid')(x_cat_conv_split_h)

    x_cat_conv_split_w = Conv2D(channel, kernel_size=1, strides=1, use_bias=False,
                                name="ca_block_conv3_" + str(name))(x_cat_conv_split_w)
    x_cat_conv_split_w = Activation('sigmoid')(x_cat_conv_split_w)

    output = multiply([input_feature, x_cat_conv_split_h])
    output = multiply([output, x_cat_conv_split_w])
    return output


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='macro')
        _val_recall = recall_score(val_targ, val_predict,average='macro')
        _val_precision = precision_score(val_targ, val_predict,average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return



class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224,3)
        self.epoch = 50
        self.batchsize = 16
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.validation_data = None
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

    def conv_bn(self, x, nb_filters, kernel_size, padding="same", strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + "_bn"
            conv_name = name + "_conv"
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(nb_filters, kernel_size, padding=padding, strides=strides, name=conv_name)(x)

        x = BatchNormalization(axis=-1, name=bn_name)(x)
        x = Activation('relu')(x)

        return x

    def bottle_block(self, input, nb_filters, padding="same", strides=(1, 1), with_conv_shortcut=False):
        k1, k2, k3 = nb_filters
        x = self.conv_bn(input, k1, (3, 3), padding=padding, strides=strides)
        x = self.conv_bn(x, k2, (5, 5), padding=padding)
        x = self.conv_bn(x, k3, (7, 7), padding=padding)

        if with_conv_shortcut:
            shortcut = self.conv_bn(input, k3, (1, 1), padding=padding, strides=strides)
            x = add([x, shortcut])
        else:
            x = add([x, input])
        return x

    def gp_3x3(self, inputs, pre_layer):
        interval = int(pre_layer / 2)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = DepthwiseConv2D((3, 3), padding='same',
                                  input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)

        # 第二路
        tower_2 = DepthwiseConv2D((3, 3), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)

        output = tensorflow.keras.layers.concatenate([tower_1, tower_2], axis=3,name=str(uuid.uuid4()))
        # output = self.channel_shuffle(output, 2)

        output = Conv2D(pre_layer // 4, 1, kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(
            output)
        output = BatchNormalization()(output)

        return output

    def gp_5x5(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=(128,128,3), kernel_initializer='uniform',
                                  kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)

        # 第四路
        tower_4 = DepthwiseConv2D((5, 1), padding='same',
                                  input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = DepthwiseConv2D((1, 5), padding='same',
                                  input_shape=(128,128,3), kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

        output = tensorflow.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
        # output = self.channel_shuffle(output, 4)

        output = Conv2D(pre_layer // 4, 1,kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay), activation='relu')(
            output)
        output = BatchNormalization()(output)

        return output

    def gp_7x7(self, inputs, pre_layer):
        interval = int(pre_layer / 4)

        normalized1_1 = Lambda(lambda x: x[:, :, :, :interval],name=str(uuid.uuid4()))(inputs)
        normalized1_2 = Lambda(lambda x: x[:, :, :, interval:2 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_3 = Lambda(lambda x: x[:, :, :, 2 * interval:3 * interval],name=str(uuid.uuid4()))(inputs)
        normalized1_4 = Lambda(lambda x: x[:, :, :, 3 * interval:],name=str(uuid.uuid4()))(inputs)

        # 第一路
        tower_1 = SeparableConv2D(interval // 4, (7, 1), padding='same',
                                  input_shape=(128,128,3),kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_1)
        tower_1 = SeparableConv2D(interval // 4, (1, 7),kernel_initializer='uniform', padding='same',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_1)

        # 第二路
        tower_2 = SeparableConv2D(interval // 4, (7, 1),kernel_initializer='uniform', padding='same',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_2)
        tower_2 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_2)

        # 第三路
        tower_3 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_3)
        tower_3 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_3)
        # 第四路
        tower_4 = SeparableConv2D(interval // 4, (7, 1), padding='same',kernel_initializer='uniform',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(normalized1_4)
        tower_4 = SeparableConv2D(interval // 4, (1, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(tower_4)

        output = tensorflow.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
        output = BatchNormalization()(output)

        return output

    def gp_block(self, inputs, in_filters):
        # 生成分组卷积模块

        # 第一路
        tower_1 = Conv2D(in_filters // 4, (1, 1), padding='same',name=str(uuid.uuid4()),kernel_initializer='uniform',
                         input_shape=(128,128,3), kernel_regularizer=regularizers.l2(self.weight_decay),
                         activation='relu')(inputs)
        tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = self.gp_3x3(inputs, in_filters)

        # 第三路
        tower_3 = self.gp_5x5(inputs, in_filters)

        # 第四路
        tower_4 = self.gp_7x7(inputs, in_filters)

        # 第四路
        # tower_5 = self.gp_9x9(inputs, in_filters)
        # output = [tower_1, tower_2, tower_3, tower_4]
        x = tf.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3, name=str(uuid.uuid4()))
        # x = self.SKBlock(tower_1, tower_2, tower_3, tower_4)
        # fused = WeightedFusion(4)(output)

        return x


    def SKBlock(self, tower_1, tower_2, tower_3, tower_4):
        merged = tf.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3, name=str(uuid.uuid4()))
        gap = GlobalAvgPool2D()(merged)
        fc = Dense( tower_1.shape[-1] //8, activation='relu')(gap)
        attn = Dense(tower_1.shape[-1] * 4, activation='softmax')(fc)  # 动态权重
        attn_b1,attn_b2, attn_b3,attn_b4 = tf.split(attn, num_or_size_splits=4, axis=1)
        output = Multiply()([tower_1, attn_b1])+Multiply()([tower_2, attn_b2])+Multiply()([tower_3, attn_b3])+Multiply()([tower_4, attn_b4])
        shortcut = Conv2D(tower_1.shape[-1], 1)(merged)
        return tf.keras.layers.Add()([output, shortcut])


    def py_block(self, inputs, in_filters, out_filters):
        # 生成金字塔卷积模块

        weight_decay = 0.0005

        x1 = SeparableConv2D(in_filters, 3,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

        x2 = SeparableConv2D(in_filters, 5,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

        x3 = SeparableConv2D(in_filters, 7,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)


        x1 = BatchNormalization()(x1)
        x2 = BatchNormalization()(x2)
        x3 = BatchNormalization()(x3)

        x1 = self.gp_block(x1, in_filters)
        x2 = self.gp_block(x2, in_filters)
        x3 = self.gp_block(x3, in_filters)

        x = tf.keras.layers.concatenate([x1, x2, x3], axis=3, name=str(uuid.uuid4()))

        x = cbam.cbam_module(x)

        x = Conv2D(out_filters, 3, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(x)
        x = BatchNormalization()(x)

        inputs = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(inputs)
        inputs = BatchNormalization()(inputs)

        x = tensorflow.keras.layers.add([inputs, x])
        x = Activation('relu')(x)

        return x


    def multi_feature(self,x):

        x3 = SeparableConv2D(64, 3, padding='same', activation='relu',  kernel_initializer='he_normal', name='conv2d_3_1')(x)

        x5 = SeparableConv2D(64, 5, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_5_1')(x)
        # x5Avg = AveragePooling2D()(x5)
        # x5Max = MaxPooling2D()(x5)
        # x5Mer = self.build_dynamic_conv_model(x5Avg,x5Max)
        # x5Mer = multiply([x5Avg,x5Max])

        x35 = concatenate([x3,x5])
        x35final = SeparableConv2D(64, 3, padding='same', activation='relu',  kernel_initializer='he_normal', name='conv2d_35_1')(x35)

        x7 = Conv2D(64, 7, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_1_1')(x)
        # x7Avg = AveragePooling2D()(x7)
        # x7Max = MaxPooling2D()(x7)
        # x7Mer = self.build_dynamic_conv_model(x7Avg,x7Max)
        # x7Mer = multiply([x7Avg,x7Max])
        x357 = concatenate([x7, x35final])
        x37final = SeparableConv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_37_1')(x357)

        x9 = SeparableConv2D(64, 9, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_19_1')(x)
        # x9Avg = AveragePooling2D()(x9)
        # x9Max = MaxPooling2D()(x9)
        # x9Mer = self.build_dynamic_conv_model(x9Avg,x9Max)
        # x9Mer = multiply([x9Max,x9Avg])

        x39 = concatenate([x37final, x9])
        x39final = SeparableConv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_39_1')(x39)

        # x11 = Conv2D(32, 11, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_111_1')(x)
        # x11Avg = AveragePooling2D()(x11)
        # x11Max = MaxPooling2D()(x11)
        # x11Mer = self.build_dynamic_conv_model(x11Avg,x11Max)
        # x11Mer = multiply([x11Avg,x11Max])

        # x311 = concatenate([x37final, x11])
        # x311final = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal', name='conv2d_311_1')(x311)
        #
        # 动态权重计算（SE注意力机制改进版）
        # def channel_attention(x1, x2,x3):
        #     gap1 = GlobalAvgPool2D()(x1)
        #     gap2 = GlobalAvgPool2D()(x2)
        #     gap3 = GlobalAvgPool2D()(x3)
        #     # gap4 = GlobalAvgPool2D()(x4)
        #     gap_concat = tf.concat([gap1, gap2,gap3], axis=-1)
        #
        #     # 权重生成网络
        #     # weights = Dense(1024, activation='relu')(gap_concat)
        #     weights = Dense(4, activation='softmax')(gap_concat)  # 两个卷积的权重系数
        #     return weights
        #
        # # 获取动态权重并加权融合
        # weights = channel_attention(x35final,x37final,x39final)
        # w1 = tf.reshape(weights[:, 0], [-1, 1, 1, 1])
        # w2 = tf.reshape(weights[:, 1], [-1, 1, 1, 1])
        # w3 = tf.reshape(weights[:, 2], [-1, 1, 1, 1])
        # # w4 = tf.reshape(weights[:, 3], [-1, 1, 1, 1])
        #
        # # 特征加权融合
        # fused = Add()([
        #     Multiply()([x35final, w1]),
        #     Multiply()([x37final, w2]),
        #     Multiply()([x39final, w3]),
        #     # Multiply()([x311final, w4])
        # ])

        fused= add([x35final,x37final,x39final])
        return fused

    # ---------- 可复用的小工具 ----------
    def convbn(x, filters, k=3, s=1, activation='relu', name=None):
        """Conv-BN-Activation 三合一套餐"""
        x = L.Conv2D(filters, k, strides=s, padding='same', use_bias=False,
                     kernel_initializer='he_normal', name=name + '_conv' if name else None)(x)
        x = L.BatchNormalization(name=name + '_bn' if name else None)(x)
        if activation:
            x = L.Activation(activation, name=name + '_act' if name else None)(x)
        return x

    def coordgate(self,x, reduction=8, name='coordgate'):
        """
        创新点 1：CoordGate 空间注意力
        在 H,W 方向分别做平均池化 -> 1D 坐标特征 -> 拼接 -> 两次 FC -> sigmoid 生成 gate
        相比 7×7 卷积，它天然携带绝对坐标信息，对“驾驶员在画面下方”这类强位置先验更敏感
        """
        b, h, w, c = K.int_shape(x)
        # 垂直坐标编码
        y_pool = L.Lambda(lambda z: K.mean(z, axis=2, keepdims=True), name=name + '_y_pool')(x)  # (B,H,1,C)
        y_vec = L.Conv2D(c // reduction, 1, activation='relu', name=name + '_y_conv1')(y_pool)
        y_vec = L.Conv2D(c, 1, activation='sigmoid', name=name + '_y_conv2')(y_vec)  # (B,H,1,C)
        # 水平坐标编码
        x_pool = L.Lambda(lambda z: K.mean(z, axis=1, keepdims=True), name=name + '_x_pool')(x)  # (B,1,W,C)
        x_vec = L.Conv2D(c // reduction, 1, activation='relu', name=name + '_x_conv1')(x_pool)
        x_vec = L.Conv2D(c, 1, activation='sigmoid', name=name + '_x_conv2')(x_vec)  # (B,1,W,C)
        # 广播相乘
        pos_gate = L.Multiply(name=name + '_gate')([y_vec, x_vec])  # (B,H,W,C)
        return pos_gate

    def fbgate(self,x, reduction=16, name='fbgate'):
        """
        创新点 2：Foreground-Background 对比门
        步骤：
        1. 用 1×1 卷积在线生成 soft 前景概率图 M (0~1)
        2. 分别计算前景/背景统计向量：全局平均池化加权
        3. 拼接后做 FC -> sigmoid 得到通道 gate
        作用：把“身体/手机”等前景通道拉高，把“车窗、道路”等背景通道压低
        """
        b, h, w, c = K.int_shape(x)
        # 1. 软前景 mask
        mask = L.Conv2D(1, 1, activation='sigmoid', name=name + '_mask')(x)  # (B,H,W,1)
        # 2. 前景、背景统计
        fg_vec = L.GlobalAveragePooling2D(name=name + '_fg_pool')(x * mask)  # (B,C)
        bg_vec = L.GlobalAveragePooling2D(name=name + '_bg_pool')(x * (1 - mask))
        concat = L.Concatenate(name=name + '_concat')([fg_vec, bg_vec])  # (B,2C)
        # 3. FC 生成通道 gate
        z = L.Dense(c // reduction, activation='relu', name=name + '_fc1')(concat)
        ch_gate = L.Dense(c, activation='sigmoid', name=name + '_fc2')(z)  # (B,C)
        ch_gate = L.Reshape((1, 1, c), name=name + '_reshape')(ch_gate)
        return ch_gate

    def dual_attention_block(self,x, name='da'):
        """
        创新点 3：互补门融合双通道注意力
        空间支路：CoordGate
        通道支路：FBGate
        融合：空间 gate 先乘特征，通道 gate 再乘，最后与原特征残差相加
        实验发现这种串行+残差比简单 add / multiply 更能抑制背景杂波
        """
        # 空间注意力
        sp_gate = self.coordgate(x, name=name + '_coord')
        x_s = L.Multiply(name=name + '_sp_mult')([x, sp_gate])
        # 通道注意力
        ch_gate = self.fbgate(x_s, name=name + '_fb')
        x_sc = L.Multiply(name=name + '_ch_mult')([x_s, ch_gate])
        # 残差
        out = L.Add(name=name + '_res')([x, x_sc])
        return out

    def build_model(self):
        from keras_cv_attention_models import efficientvit, swin_transformer_v2, repvit, mobilevit, convnext, uniformer, iformer, \
            caformer, davit, maxvit, fastvit, flexivit, efficientformer, gpvit, gcvit, tinyvit, pvt, levit, coat, edgenext, resnext, \
            lcnet, efficientnet, halonet, efficientdet, fbnetv3, meta_transformer, nfnets, maxvit,fastervit
        # model = keras_cv_attention_models.uniformer.Uniformer(input_shape=(224, 224, 3), pretrained=None)
        # model = keras_cv_attention_models.uniformer.Uniformer(input_shape=(224, 224, 3), pretrained=None)
        # model = repvit.RepViT(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3(input_shape=(224, 224, 3),pretrained=None)
        # model = uniformer.Uniformer(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientvit.EfficientViT_B0(input_shape=(224, 224, 3),pretrained="imagenet")
        # model = convnext.ConvNeXtV2(input_shape=(224, 224, 3),pretrained="imagenet")
        # model = efficientformer.EfficientFormer(input_shape=(224, 224, 3),pretrained=Nonde)
        model = gpvit.GPViT_L1(input_shape=(224, 224, 3), pretrained="imagenet")        # model = caformer.ConvFormerS18(input_shape=(224, 224, 3),pretrained=None)
        # model = fastvit.FastViT(input_shape=(224, 224, 3),pretrained=None)
        # model = flexivit.FlexiViT(input_shape=(224, 224, 3),pretrained=None)
        # model = davit.DaViT(input_shape=(224, 224, 3),pretrained=None)
        # model = tinyvit.TinyViT(input_shape=(224, 224, 3),pretrained=None)
        # model = cmt.CMTTiny (input_shape=(224, 224, 3),pretrained=None)
        # model = lcnet.LCNet(input_shape=(224, 224, 3),pretrained=None)
        # model = EfficientFormerV2(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientnet.EfficientNetV2(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientnet.EfficientNetV1(input_shape=(224, 224, 3),pretrained=None)
        # model = mobilevit.MobileViT_V2(input_shape=(224, 224, 3),pretrained=None)
        # model = resnext.ResNeXt50(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3D(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3G(input_shape=(224, 224, 3),pretrained=None)
        # model = fbnetv3.FBNetV3B(input_shape=(224, 224, 3),pretrained=None)
        # model = pvt.PyramidVisionTransformerV2(input_shape=(224, 224, 3),pretrained="imagenet")
        # model = mobilevit.MobileViT_S(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientformer.EfficientFormerL1(input_shape=(224, 224, 3),pretrained=None)
        # model = efficientformer.EfficientFormerV2L(input_shape=(224, 224, 3),pretrained=None)
        #
        x = model.layers[-2].output
        outputs = Dense(self.classes, activation="softmax",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=model.inputs, outputs=outputs)
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
        X_train = np.load(r'C:\Users\Administrator\Desktop\four\driver_feature_auc_four_split/train/images1.npy')

        X_train = X_train.reshape([-1, 224, 224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_train)
        # X_train = X_train.astype(np.float32)

        y_train = np.load(r'C:\Users\Administrator\Desktop\four\driver_feature_auc_four_split/train/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_train)

        X_valid = np.load(r'C:\Users\Administrator\Desktop\four\driver_feature_auc_four_split/test/images1.npy')

        X_valid = X_valid.reshape([-1, 224, 224, 3])
        np.random.seed(2025)
        np.random.shuffle(X_valid)
        # X_valid = X_valid.astype(np.float32)

        y_valid = np.load(r'C:\Users\Administrator\Desktop\four\driver_feature_auc_four_split/test/labels.npy')
        np.random.seed(2025)
        np.random.shuffle(y_valid)

        # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=2025)
        print(X_train.shape)
        print(y_train.shape)
        print(X_valid.shape)
        print(y_valid.shape)

        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy',  # 损失函数
                           metrics=['accuracy'])  # 指标

        metrics = Metrics()
        self.model.validation_data = (X_valid, y_valid)

        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batchsize,
                                 epochs=self.epoch,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=(X_valid, y_valid),
                                 callbacks=[metrics]
                                 )
        # 靠谱方式
        data_path_abs = '../driver'
        img_list_all = os.listdir(data_path_abs)
        for key, v in enumerate(img_list_all):
            input_img1 = cv2.imread(data_path_abs + "/" + v)
            input_img1 = cv2.resize(input_img1, (224, 224))
            input_img1 = np.expand_dims(input_img1, axis=0)
            # sobel_image = sobel_image - np.mean(sobel_image, axis=0)
            predictions = self.model.predict(input_img1)
            #     # 获取最后一层卷积层的输出
            last_conv_layer = self.model.get_layer('conv_pw_11_relu')
            grad_model = Model([self.model.inputs], [last_conv_layer.output, self.model.output])
            #     # 计算类别的梯度
            with tf.GradientTape() as tape:
                conv_layer_output, preds = grad_model(input_img1)
                class_channel = preds[0][np.argmax(preds[0])]
            # 计算梯度
            grads = tape.gradient(class_channel, conv_layer_output)
            # 计算权重
            print(v, end="------------")
            print(preds[0], end="------------np.argmax(preds[0]):")
            print(np.argmax(preds[0]), end="------------class_channel:")
            # class_channel = preds[0][np.argmax(preds[0])]
            print(class_channel)
            print("==================================================")

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            # pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            # heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_layer_output), axis=-1)
            heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
            # feature_map_sum = sum(ele for ele in feature_map_combination)
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.reduce_max(heatmap)

            # 重塑热力图并将其缩放到与原始图像相同的大小
            heatmap = np.squeeze(heatmap)
            # 颜色映射
            gbkInput = cv2.imread(data_path_abs + "/" + v)
            gbkInput = cv2.resize(gbkInput, (224, 224))

            feature_map_sum1 = cv2.resize(heatmap, (224, 224))
            # 将热力图转换为RGB格式
            feature_map_sum = np.uint8(255 * feature_map_sum1)
            feature_map_sum[feature_map_sum < 80] = 0
            # 将热利用应用于原始图像
            feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
            # 　这里的热力图因子是０.４
            superimposed_img = feature_map_sum * 0.4 + gbkInput
            cv2.imwrite("person{}".format(v + ".jpg"), superimposed_img)
            # cv2.imwrite( "person{}".format(v+".eps"), superimposed_img)
            plt.savefig("{}".format("person" + str(uuid.uuid4()) + ".eps"))

            print("热力图完成")
        yPred = []
        y = []
        for i in range(len(X_test)):
            image = np.expand_dims(X_test[i], axis=0)
            predictions = self.model.predict(image)
            top_class_index = tf.argmax(predictions, axis=-1)
            # top_class_probability = predictions[0][top_class_index]
            yPred.append([int(top_class_index.numpy())])
            x_real = [k for k, v in enumerate(y_test[i]) if v == 1]
            y.append(x_real)
        draw_confu(y, yPred, name='train')

        _val_f1 = f1_score(y, yPred, average='micro')
        _val_recall = recall_score(y, yPred, average='micro')

        print("draw compliment" + str(_val_f1) + "*---" + str(_val_recall))

        fer_json = self.model.to_json()
        with open("xiaorong_cam.json", "w") as json_file:
            json_file.write(fer_json)
        self.model.save_weights("xiaorong_cam.h5")
        print("Saved model to disk")

def draw_confu(y, y_pred, name=''):
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
    plt.xticks(fontsize=10)  # 设置x轴刻度字体大小为12
    plt.yticks(fontsize=10)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=32)
    plt.ylabel('Actual Label', fontsize=28)
    plt.xlabel('Predicted Label', fontsize=28)
    plt.savefig('./result_%s.eps' % (name))
    plt.savefig('./result_%s.svg' % (name))
    plt.savefig('./result_%s.jpg' % (name))

def evaluate(model, X, Y):
    accuracy = model.evaluate(X, Y)
    return accuracy[0]

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
    import tensorflow as tf
    from tensorflow.keras import backend as K
    K.clear_session()
    tf.config.run_functions_eagerly(True)

    import random

    random.seed(2025)
    import numpy as np
    np.random.seed(2025)
    tf.random.set_seed(2025)
    import os

    os.environ['PYTHONHASHSEED'] = '2025'

    # 不使用gpu则开启这一行代码
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(tf.test.is_gpu_available())
    fer_model = FerModel()
    print(tf.test.is_gpu_available())
