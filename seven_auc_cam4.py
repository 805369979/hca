import os
import cv2
import math
import uuid
import time
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense,
                                     PReLU, Input, BatchNormalization, GlobalMaxPooling2D, SeparableConv2D,
                                     LeakyReLU, Concatenate, Lambda, Flatten, InputLayer)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, AveragePooling2D, add
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import concatenate, multiply

import cbam
from data_loader import SingleDirectoryDataLoader

unique_random_number = uuid.uuid4()


class Metrics(Callback):
    model = None
    val_generator = None

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        if self.val_generator is None:
            return
        # 分批预测，避免OOM
        val_predict_list = []
        val_targ_list = []
        for i in range(len(self.val_generator)):
            x, y = self.val_generator[i]
            pred = self.model.predict(x, verbose=0)
            val_predict_list.append(pred)
            val_targ_list.append(y)
        val_predict = np.concatenate(val_predict_list, axis=0)
        val_targ = np.concatenate(val_targ_list, axis=0)

        val_predict_labels = np.argmax(val_predict, axis=1)
        if len(val_targ.shape) > 1 and val_targ.shape[1] > 1:
            val_targ_labels = np.argmax(val_targ, axis=1)
        else:
            val_targ_labels = val_targ.astype(int)

        _val_f1 = f1_score(val_targ_labels, val_predict_labels, average='weighted', zero_division=0)
        _val_recall = recall_score(val_targ_labels, val_predict_labels, average='weighted', zero_division=0)
        _val_precision = precision_score(val_targ_labels, val_predict_labels, average='weighted', zero_division=0)

        print(" — F1-score: %.2f%% — Precision: %.2f%% — Recall: %.2f%%" %
              (_val_f1 * 100, _val_precision * 100, _val_recall * 100))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return


class FerModel(object):
    def __init__(self):
        self.x_shape = (224, 224, 3)
        self.epoch = 30
        self.batchsize = 32
        self.weight_decay = 0.0005
        self.classes = 10
        self.model = self.build_model()
        self.call_backs = self.get_call_backs()
        start = time.time()
        self.validation_data = None
        self.history = self.train()
        end = time.time()
        print(f'训练共耗时{round(end - start, 2)}s')

    @staticmethod
    def get_call_backs():
        call_backs = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
        ]
        print("调用回调函数!!!")
        return call_backs

    def global_context_attention(self, input_tensor, reduction_ratio=16):
        """全局上下文注意力"""
        channels = input_tensor.shape[-1]
        gap = GlobalAveragePooling2D()(input_tensor)
        gap = layers.Reshape((1, 1, channels))(gap)
        context_weights = layers.Dense(channels // reduction_ratio, activation='relu')(gap)
        context_weights = layers.Dense(channels, activation='sigmoid')(context_weights)
        return layers.Multiply()([input_tensor, context_weights])

    def sobel_branch(self, inp):
        """Sobel 边缘检测分支"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel = np.stack([sobel_x] * 3, axis=-1)[..., None]
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        kernely = np.stack([sobel_y] * 3, axis=-1)[..., None]

        edge = layers.Conv2D(1, 3, padding='same', use_bias=False,
                             weights=[kernel], trainable=False, name='sobel')(inp)
        edgey = layers.Conv2D(1, 3, padding='same', use_bias=False,
                             weights=[kernely], trainable=False, name='sobel1')(inp)
        edge1 = layers.add([edge, edgey])

        edge = SeparableConv2D(64, (9, 9), activation='relu', padding='same')(edge1)
        edge = layers.MaxPooling2D()(edge)
        edge1 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(edge)
        edge1 = SeparableConv2D(64, (5, 5), activation='relu', padding='same')(edge1)
        edge1 = SeparableConv2D(64, (7, 7), activation='relu', padding='same')(edge1)
        edge = add([edge1, edge])

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

        edge = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(edge)
        edge = layers.MaxPooling2D()(edge)
        edge = cbam.cbam_module(edge)
        return edge

    def dynamic_weight_fusion(self, base, scale):
        """2路 softmax 加权融合（与单视角完全一致）"""
        weights = layers.Concatenate()([base, scale])
        weights = layers.Dense(64, activation='relu')(weights)
        weights = layers.Dense(2, activation='softmax')(weights)
        weighted_base = layers.Multiply()([base, weights[:, 0:1]])
        weighted_scale = layers.Multiply()([scale, weights[:, 1:2]])
        return layers.Add()([weighted_base, weighted_scale])

    def build_model(self):
        """单视角模型：与 seven_auc copy.py 的 build_model 完全一致"""
        inputs = Input(shape=self.x_shape)

        # Sobel 边缘分支
        soble = self.sobel_branch(inputs)

        # MobileNet 主干（与原始单视角一致：input_tensor=inputs，取 conv_pw_11_relu）
        base = MobileNet(include_top=False, weights='imagenet', input_tensor=inputs)
        base = base.get_layer('conv_pw_11_relu')

        # 相乘融合
        x = multiply([base.output, soble])

        # GAP
        x = GlobalAveragePooling2D()(x)
        base_gap = GlobalAveragePooling2D()(base.output)

        # 2路 softmax 加权融合
        x = self.dynamic_weight_fusion(base_gap, x)

        # 输出层
        outputs = Dense(self.classes, activation="softmax",
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        model_ = Model(inputs=inputs, outputs=outputs)
        model_.summary()
        return model_

    def train(self):
        """只训练和测试 Cam4 视角"""
        # 数据路径（根据实际路径修改）
        train_dir = "D:\\数据集\\100-Driver\\splittrain\\Cam4"
        test_dir = "D:\\数据集\\100-Driver\\splittest\\Cam4"

        batch_size = self.batchsize
        sample = 2000

        # 训练集生成器（单目录）
        train_generator = SingleDirectoryDataLoader(
            data_dir=train_dir,
            batch_size=batch_size, sampleNum=sample,
            img_size=self.x_shape[:2],
            shuffle=True, is_training=True
        )

        # 验证集生成器（单目录）
        val_generator = SingleDirectoryDataLoader(
            data_dir=test_dir,
            batch_size=batch_size, sampleNum=sample,
            img_size=self.x_shape[:2],
            shuffle=False, is_training=False
        )

        num_classes = train_generator.get_num_classes()
        print(f"类别数: {num_classes}, 类别名: {train_generator.get_class_names()}")
        print(f"训练样本数: {len(train_generator.image_paths)}")
        print(f"验证样本数: {len(val_generator.image_paths)}")

        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.95)
        self.model.compile(optimizer=momentum_optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        metrics = Metrics()
        metrics.val_generator = val_generator

        history = self.model.fit(
            train_generator,
            epochs=self.epoch,
            verbose=1,
            validation_data=val_generator,
            callbacks=[metrics] + self.call_backs
        )
        return history


def Main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    K.clear_session()
    import random
    random.seed(2025)
    np.random.seed(2025)
    tf.random.set_seed(2025)

    fer = FerModel()


if __name__ == '__main__':
    Main()
