import cv2
from keras_cv_attention_models import mobilevit
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
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import concatenate, Conv2DTranspose, ZeroPadding2D, multiply, UpSampling2D, Add, \
    Reshape, Conv1D, ReLU, GlobalAvgPool2D, Multiply, GlobalMaxPool2D

import cbam
import uuid
unique_random_number = uuid.uuid4()

from tensorflow.keras import layers as L
from tensorflow.keras import backend as K


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
        self.epoch = 30
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

    def gp_3x3(self, inputs, pre_layer):
        interval = int(pre_layer )

        tower_4 = SeparableConv2D(interval, (3, 3), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(inputs)
        output = BatchNormalization()(tower_4)

        return output

    def gp_5x5(self, inputs, pre_layer):
        interval = int(pre_layer)

        tower_4 = SeparableConv2D(interval,(5, 5), padding='same',
                                  input_shape=self.x_shape, kernel_initializer='uniform',kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(inputs)

        output = BatchNormalization()(tower_4)

        return output

    def gp_7x7(self, inputs, pre_layer):
        interval = int(pre_layer)

        tower_4 = SeparableConv2D(interval, (7, 7), padding='same',kernel_initializer='uniform',
                                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
                                  activation='relu')(inputs)
        output = BatchNormalization()(tower_4)
        return output

    def channel_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 重新排序维度[b,h,w,c]==>[b,c,h,w]
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])  # perm代表重新排序的轴
        # 重塑特征图尺寸[b,c,h,w]==>[b,c,h*w]
        x_reshape = tf.reshape(x, shape=[-1, c, h * w])

        # 重新排序维度[b,c,h*w]==>[b,h*w,c]
        x_reshape_trans = tf.transpose(x_reshape, perm=[0, 2, 1])  # 指定需要交换的轴
        # 矩阵相乘
        x_mutmul = x_reshape_trans @ x_reshape
        # 经过softmax归一化权重
        x_mutmul = tf.nn.softmax(x_mutmul)

        # reshape后的特征图与归一化权重矩阵相乘[b,x,h*w]
        x = x_reshape @ x_mutmul
        # 重塑形状[b,c,h*w]==>[b,c,h,w]
        x = tf.reshape(x, shape=[-1, c, h, w])
        # 重新排序维度[b,c,h,w]==>[b,h,w,c]
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x = x * gama

        # 输入和输出特征图叠加
        x = add([x, inputs])

        return x

    # （2）位置注意力
    def position_attention(self,inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c//8]
        x1 = SeparableConv2D(filters=c // 8, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c//8]==>[b,c//8,h,w]
        x1_trans = tf.transpose(x1, perm=[0, 3, 1, 2])
        # 重塑特征图尺寸[b,c//8,h,w]==>[b,c//8,h*w]
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c // 8, h * w])
        # 调整维度排序[b,c//8,h*w]==>[b,h*w,c//8]
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape, perm=[0, 2, 1])
        # 矩阵相乘
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        # 经过softmax归一化权重
        x1_mutmul = tf.nn.softmax(x1_mutmul)

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c]
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c]==>[b,c,h,w]
        x2_trans = tf.transpose(x2, perm=[0, 3, 1, 2])
        # 重塑尺寸[b,c,h,w]==>[b,c,h*w]
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c, h * w])

        # 调整x1_mutmul的轴，和x2矩阵相乘
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2, 1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans

        # 重塑尺寸[b,c,h*w]==>[b,c,h,w]
        x2_mutmul = tf.reshape(x2_mutmul, shape=[-1, c, h, w])
        # 轴变换[b,c,h,w]==>[b,h,w,c]
        x2_mutmul = tf.transpose(x2_mutmul, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x2_mutmul = x2_mutmul * gama

        # 输入和输出叠加
        x = add([x2_mutmul, inputs])
        return x

    # （3）DANet网络架构
    def danet(self,inputs):
        # 输入分为两个分支
        x1 = self.channel_attention(inputs)  # 通道注意力
        x2 = self.position_attention(inputs)  # 位置注意力

        # 叠加两个注意力的结果
        x = add([x1, x2])
        return x

    def gp_block(self, inputs, in_filters):
        # 生成分组卷积模块

        # 第一路
        # tower_1 = Conv2D(in_filters, (1, 1), padding='same',name=str(uuid.uuid4()),kernel_initializer='uniform',
        #                  input_shape=self.x_shape, kernel_regularizer=regularizers.l2(self.weight_decay),
        #                  activation='relu')(inputs)
        # tower_1 = BatchNormalization()(tower_1)

        # 第二路
        tower_2 = self.gp_3x3(inputs, in_filters)

        # 第三路
        tower_3 = self.gp_5x5(inputs, in_filters)

        # 第四路
        tower_4 = self.gp_7x7(inputs, in_filters)

        # 第四路
        # tower_5 = self.gp_9x9(inputs, in_filters)
        # output = [ tower_2, tower_3, tower_4]

        output = tf.keras.layers.concatenate([ tower_2, tower_3, tower_4], axis=3,name=str(uuid.uuid4()))
        # output = WeightedFusion(3)(output)
        return output

    import tensorflow as tf
    from tensorflow.keras.layers import (
        GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply,
        Reshape, Activation, Conv2D, Concatenate, Add,
        AveragePooling2D, UpSampling2D, Flatten
    )
    from tensorflow.keras import backend as K

    def channel_attention(self,input_tensor, reduction_ratio=16):
        """通道注意力机制实现"""
        channels = input_tensor.shape[-1]

        # 全局平均池化和最大池化
        gap = GlobalAveragePooling2D()(input_tensor)
        gmp = GlobalMaxPooling2D()(input_tensor)

        # 共享权重处理两种池化结果
        def shared_fc(x):
            x = Dense(channels // reduction_ratio, activation='relu', use_bias=False)(x)
            x = Dense(channels, activation='sigmoid', use_bias=False)(x)
            return Reshape((1, 1, channels))(x)

        gap_weights = shared_fc(gap)
        gmp_weights = shared_fc(gmp)

        # 合并权重并应用
        channel_weights = Activation('sigmoid')(Add()([gap_weights, gmp_weights]))
        return Multiply()([input_tensor, channel_weights])

    def spatial_attention(self,input_tensor, kernel_size=7):
        """空间注意力机制实现"""
        # 通道维度池化
        avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)

        # 拼接并卷积
        concat = Concatenate()([avg_pool, max_pool])
        spatial_weights = Conv2D(1, kernel_size=kernel_size,
                                 padding='same', activation='sigmoid')(concat)

        return Multiply()([input_tensor, spatial_weights])

    def cbam_attention(self,input_tensor, reduction_ratio=16, kernel_size=7):
        """CBAM混合注意力机制"""
        x = self.channel_attention(input_tensor, reduction_ratio)
        x = self.spatial_attention(x, kernel_size)
        return x

    def eca_attention(self,input_tensor, kernel_size=3):
        """ECA高效通道注意力"""
        channels = input_tensor.shape[-1]

        # 全局平均池化
        gap = GlobalAveragePooling2D()(input_tensor)
        gap = Reshape((1, 1, channels))(gap)

        # 一维卷积学习通道间关系
        eca_weights = Conv2D(1, kernel_size=(1, kernel_size),
                             padding='same', use_bias=False)(gap)
        eca_weights = Activation('sigmoid')(eca_weights)

        return Multiply()([input_tensor, eca_weights])

    def self_attention_2d(self,input_tensor):
        """二维自注意力机制"""
        # 获取输入形状
        batch_size = tf.shape(input_tensor)[0]
        height = input_tensor.shape[1]
        width = input_tensor.shape[2]
        channels = input_tensor.shape[-1]

        # 重塑为序列形式
        x_flat = Reshape((height * width, channels))(input_tensor)

        # 计算注意力分数
        attention_scores = Dense(channels)(x_flat)
        attention_scores = Activation('softmax')(attention_scores)

        # 应用注意力
        attended = tf.matmul(attention_scores, x_flat, transpose_a=True)
        attended = Reshape((height, width, channels))(attended)

        return attended

    def multi_head_self_attention(self,input_tensor, num_heads=8):
        """多头自注意力机制"""
        channels = input_tensor.shape[-1]
        height = input_tensor.shape[1]
        width = input_tensor.shape[2]
        head_dim = channels // num_heads

        # 重塑为序列
        x_flat = Reshape((height * width, channels))(input_tensor)

        # 线性变换得到Q, K, V
        q = Dense(channels)(x_flat)
        k = Dense(channels)(x_flat)
        v = Dense(channels)(x_flat)

        # 重塑为多头格式
        q_reshaped = Reshape((height * width, num_heads, head_dim))(q)
        k_reshaped = Reshape((height * width, num_heads, head_dim))(k)
        v_reshaped = Reshape((height * width, num_heads, head_dim))(v)

        # 转置以分离批次和头维度
        q_transposed = tf.transpose(q_reshaped, perm=[0, 2, 1, 3])
        k_transposed = tf.transpose(k_reshaped, perm=[0, 2, 1, 3])
        v_transposed = tf.transpose(v_reshaped, perm=[0, 2, 1, 3])

        # 计算注意力分数
        scores = tf.matmul(q_transposed, k_transposed, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(head_dim, tf.float32))
        weights = Activation('softmax')(scores)

        # 应用注意力权重
        output = tf.matmul(weights, v_transposed)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = Reshape((height * width, channels))(output)
        output = Reshape((height, width, channels))(output)

        return output

    def dual_attention(self,input_tensor, reduction_ratio=16):
        """双重注意力机制"""
        # 位置注意力
        pos_attention = self.self_attention_2d(input_tensor)

        # 通道注意力
        channel_att =self.channel_attention(input_tensor, reduction_ratio)

        # 融合两种注意力
        return Add()([pos_attention, channel_att])

    def pyramid_attention(self,input_tensor, reduction_ratio=16):
        """金字塔注意力机制"""
        channels = input_tensor.shape[-1]

        # 不同尺度的池化
        pool1 = AveragePooling2D(pool_size=1)(input_tensor)
        pool2 = AveragePooling2D(pool_size=2)(input_tensor)
        pool4 = AveragePooling2D(pool_size=4)(input_tensor)

        # 上采样到相同尺寸
        pool2 = UpSampling2D(size=2)(pool2)
        pool4 = UpSampling2D(size=4)(pool4)

        # 融合多尺度特征
        fused = Add()([pool1, pool2, pool4])

        # 通道注意力
        gap = GlobalAveragePooling2D()(fused)
        weights = Dense(channels // reduction_ratio, activation='relu')(gap)
        weights = Dense(channels, activation='sigmoid')(weights)
        weights = Reshape((1, 1, channels))(weights)

        return Multiply()([input_tensor, weights])

    def coordinate_attention(self,input_tensor, reduction_ratio=32):
        """坐标注意力机制"""
        channels = input_tensor.shape[-1]
        height = input_tensor.shape[1]
        width = input_tensor.shape[2]

        # 水平方向注意力
        x_h = tf.reduce_mean(input_tensor, axis=2)
        x_h = Dense(channels // reduction_ratio, activation='relu')(x_h)
        x_h = Dense(channels, activation='sigmoid')(x_h)
        x_h = Reshape((1, height, channels))(x_h)

        # 垂直方向注意力
        x_w = tf.reduce_mean(input_tensor, axis=1)
        x_w = Dense(channels // reduction_ratio, activation='relu')(x_w)
        x_w = Dense(channels, activation='sigmoid')(x_w)
        x_w = Reshape((width, 1, channels))(x_w)

        # 合并坐标注意力
        coord_weights = Multiply()([x_h, x_w])

        return Multiply()([input_tensor, coord_weights])

    def global_context_attention(self,input_tensor, reduction_ratio=16):
        """全局上下文注意力"""
        channels = input_tensor.shape[-1]

        # 全局平均池化
        gap = GlobalAveragePooling2D()(input_tensor)
        gap = Reshape((1, 1, channels))(gap)

        # 学习上下文权重
        context_weights = Dense(channels // reduction_ratio, activation='relu')(gap)
        context_weights = Dense(channels, activation='sigmoid')(context_weights)

        return Multiply()([input_tensor, context_weights])

    def squeeze_excitation_attention(self,input_tensor, reduction_ratio=16):
        """Squeeze-and-Excitation注意力"""
        channels = input_tensor.shape[-1]

        # Squeeze: 全局平均池化
        squeeze = GlobalAveragePooling2D()(input_tensor)
        squeeze = Reshape((1, 1, channels))(squeeze)

        # Excitation: 两层全连接
        excitation = Dense(channels // reduction_ratio, activation='relu')(squeeze)
        excitation = Dense(channels, activation='sigmoid')(excitation)

        return Multiply()([input_tensor, excitation])

    def ca_block(self,input_feature, ratio=16, name=""):

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
    def py_block(self, inputs, in_filters, out_filters):
        # 生成金字塔卷积模块

        x = Conv2D(in_filters, 1,kernel_initializer='uniform', padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), name=str(uuid.uuid4()), activation='relu')(inputs)

        x = BatchNormalization()(x)

        x = self.gp_block(x, in_filters)

        # x = self.ca_block(x)
        x = cbam.cbam_module(x)

        x = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(x)
        x = BatchNormalization()(x)

        inputs = Conv2D(out_filters, 1, padding='same',kernel_initializer='uniform', kernel_regularizer=regularizers.l2(self.weight_decay),name=str(uuid.uuid4()))(inputs)
        inputs = BatchNormalization()(inputs)

        x = tensorflow.keras.layers.add([inputs, x])
        x = Activation('relu')(x)

        return x
    def multi_feature(self, x):
        """
        改进型多尺度特征提取模块
        输入: x - 输入特征图
        输出: x39 - 融合后的多尺度特征
        """
        # 新增批归一化层提升训练稳定性
        def conv_block(input_tensor, filters, kernel_size, name):
            tensor = SeparableConv2D(
                filters,
                kernel_size,
                padding='same',
                activation='relu',
                
                # kernel_initializer='he_normal',
                name=f'conv_{kernel_size}_{name}'
            )(input_tensor)
            tensor = BatchNormalization(axis=-1, name=f'bn_{kernel_size}_{name}')(tensor)
            return tensor

        # 各分支并行处理
        x3 = conv_block(x, 64, 3, '1')
        x5 = conv_block(x, 64, 5, '1')
        x7 = conv_block(x, 64, 7, '1')
        x9 = conv_block(x, 64, 9, '1')

        # 多级特征融合
        x35 = concatenate([x3, x5])
        x35final = conv_block(x35, 64, 3, '35_1')

        x357 = concatenate([x7, x35final])
        x37final = conv_block(x357, 64, 3, '37_1')

        x59 = concatenate([x37final, x9])
        x59final = conv_block(x59, 64, 3, '59_1')

        # 最终特征融合
        x39 = add([x59final, x35final, x37final])
        return x39


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

        soble = sobel_branch(inputs)

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

        X_train = np.load('./driver_feature_small_driver100/train/images.npy')
        X_train = X_train.astype(np.float16)
        X_train = X_train.reshape([-1, 224, 224, 3])
        np.random.seed(42)
        np.random.shuffle(X_train)

        y_train = np.load('./driver_feature_small_driver100/train/labels.npy')
        np.random.seed(42)
        np.random.shuffle(y_train)

        X_val = np.load('./driver_feature_small_driver100/val/images.npy')
        X_val = X_val.astype(np.float16)
        X_val = X_val.reshape([-1, 224, 224, 3])
        np.random.seed(42)
        np.random.shuffle(X_val)

        y_val = np.load('./driver_feature_small_driver100/val/labels.npy')
        np.random.seed(42)
        np.random.shuffle(y_val)

        X_test = np.load('./driver_feature_small_driver100/test/images.npy')
        X_test = X_test.astype(np.float16)
        X_test = X_test.reshape([-1, 224, 224, 3])
        np.random.seed(42)
        np.random.shuffle(X_test)

        y_test = np.load('./driver_feature_small_driver100/test/labels.npy')
        np.random.seed(42)
        np.random.shuffle(y_test)

        print(X_train.shape)
        print(y_train.shape)
        # X_train = np.load('../kaggel_7_2_1/sfd/train/images.npy')
        # X_train = X_train.astype(np.float32)
        # X_train = X_train.reshape([-1, 224, 224, 3])
        # np.random.seed(42)
        # np.random.shuffle(X_train)
# 
        # y_train = np.load('../kaggel_7_2_1/sfd/train/labels.npy')
        # np.random.seed(42)
        # np.random.shuffle(y_train)
# 
    #    X_val = np.load('../kaggel_7_2_1/sfd/val/images.npy')
        # X_val = X_val.astype(np.float32)
    #    X_val = X_val.reshape([-1, 224, 224, 3])
        # np.random.seed(42)
        # np.random.shuffle(X_val)
# 
        # y_val = np.load('../kaggel_7_2_1/sfd/val/labels.npy')
        # np.random.seed(42)
        # np.random.shuffle(y_val)
# 
        # X_test = np.load('../kaggel_7_2_1/sfd/test/images.npy')
        # X_test = X_test.astype(np.float32)
        # X_test = X_test.reshape([-1, 224, 224, 3])
        # np.random.seed(42)
       # np.random.shuffle(X_test)
# 
        # y_test = np.load('../kaggel_7_2_1/sfd/test/labels.npy')
        # np.random.seed(42)
        # np.random.shuffle(y_test)
        # import numpy as np
        # X_train = np.load('C:/Users/Administrator/Desktop/four/driver_feature_auc_four_split/train/images1.npy')
        #
        # X_train = X_train.reshape([-1, 224, 224, 3])
        # np.random.seed(2025)
        # np.random.shuffle(X_train)
        # # X_train = X_train.astype(np.float32)
        #
        # y_train = np.load('C:/Users/Administrator/Desktop/four/driver_feature_auc_four_split/train/labels.npy')
        # np.random.seed(2025)
        # np.random.shuffle(y_train)
        #
        # X_test = np.load('C:/Users/Administrator/Desktop/four/driver_feature_auc_four_split/test/images1.npy')
        #
        # X_test = X_test.reshape([-1, 224, 224, 3])
        # np.random.seed(2025)
        # np.random.shuffle(X_test)
        # # X_valid = X_valid.astype(np.float32)
        #
        # y_test = np.load('C:/Users/Administrator/Desktop/four/driver_feature_auc_four_split/test/labels.npy')
        # np.random.seed(2025)
        # np.random.shuffle(y_test)

        print(X_train.shape)
        print(y_train.shape)

        # 创建Momentum优化器
        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0049, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy',  # 损失函数
                           metrics=['accuracy'])  # 指标
        metrics = Metrics()
        self.model.validation_data = (X_test, y_test)
        # training the model
        history = self.model.fit(X_train, y_train,
                                 batch_size=self.batchsize,
                                 epochs=self.epoch,
                                 verbose=1,
                                 shuffle=True,
                                 validation_data=(X_test, y_test),
                                 callbacks=[self.call_backs]
                                 )

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