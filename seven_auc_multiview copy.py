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
import cbam
import tensorflow as tf
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import concatenate, multiply

import data_loader
from data_loader import MultiDirectoryDataLoader

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

    @staticmethod
    def get_call_backs():
        call_backs = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2),
            TensorBoard('./logs/balanced_data_model'),
        ]
        print("调用回调函数!!!")
        return call_backs

    def global_context_attention(self, inputs, view_idx=0):
        """全局上下文注意力（名称加 view_idx 后缀）"""
        p = f'v{view_idx}_'
        x_gap = GlobalAveragePooling2D(name=f'{p}gca_gap')(inputs)
        x_gap = layers.Reshape((1, 1, -1), name=f'{p}gca_reshape')(x_gap)
        x_gap = layers.Conv2D(inputs.shape[-1] // 4, (1, 1), activation='relu', name=f'{p}gca_c1')(x_gap)
        x_gap = layers.Conv2D(inputs.shape[-1], (1, 1), activation='sigmoid', name=f'{p}gca_c2')(x_gap)
        return layers.Multiply(name=f'{p}gca_mul')([inputs, x_gap])

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

    def dynamic_weight_fusion(self, base, scale, view_idx=0):
        """与单视角完全一致的2路softmax加权融合（名称加 view_idx 后缀）"""
        p = f'v{view_idx}_'
        weights = layers.Concatenate(name=f'{p}f_concat')([base, scale])
        weights = layers.Dense(64, activation='relu', name=f'{p}f_d1')(weights)
        weights = layers.Dense(2, activation='softmax', name=f'{p}f_d2')(weights)
        weighted_base = layers.Multiply(name=f'{p}f_w_base')([base, weights[:, 0:1]])
        weighted_scale = layers.Multiply(name=f'{p}f_w_scale')([scale, weights[:, 1:2]])
        return layers.Add(name=f'{p}f_add')([weighted_base, weighted_scale])

    def build_single_view_branch(self, inputs, view_idx, mobilenet_fn):
        """
        单视角子网络：输入图片直接经过 sobel 分支 + 共享 MobileNet → 相乘 → GAP
        与单视角代码 seven_auc copy.py 的 build_model 完全一致
        MobileNet 在 4 视角间共享权重（通过共享层列表）
        """
        p = f'v{view_idx}_'
        # 1. Sobel 边缘分支（每视角独立，名称加后缀避免冲突）
        soble = self.sobel_branch(inputs, view_idx)

        # 2. 共享 MobileNet 主干：直接遍历层列表，避免子模型带来的层名冲突
        base_output = mobilenet_fn(inputs)

        # 4. GAP
        soble = GlobalAveragePooling2D(name=f'{p}gap_x')(soble)

        base_gap = GlobalAveragePooling2D(name=f'{p}gap_base')(base_output)

        # 3. 相乘融合（特征图级别，与单视角一致）
        x = multiply([base_gap, soble], name=f'{p}mul')



        # 5. 返回两个特征（base_gap 和 multiply后的x），由最终融合处理
        return base_gap, x

    def build_model(self):
        # 4 个视角输入
        input1 = Input(shape=self.x_shape, name='view1_input')
        input2 = Input(shape=self.x_shape, name='view2_input')
        input3 = Input(shape=self.x_shape, name='view3_input')
        input4 = Input(shape=self.x_shape, name='view4_input')

        # 创建共享的 MobileNet：使用层列表方式，避免子模型带来的层名冲突
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=self.x_shape)
        # 收集从输入到 conv_pw_13_relu 的层列表
        mobilenet_layers = []
        for layer in base_model.layers:
            if isinstance(layer, InputLayer):
                continue
            mobilenet_layers.append(layer)
            if layer.name == 'conv_pw_13_relu':
                break

        def mobilenet_fn(inp):
            x = inp
            for layer in mobilenet_layers:
                x = layer(x)
            return x

        # 每个视角独立处理（MobileNet 共享），每视角返回2个特征（base_gap, x）
        base1, x1 = self.build_single_view_branch(input1, 1, mobilenet_fn)
        base2, x2 = self.build_single_view_branch(input2, 2, mobilenet_fn)
        base3, x3 = self.build_single_view_branch(input3, 3, mobilenet_fn)
        base4, x4 = self.build_single_view_branch(input4, 4, mobilenet_fn)

        # 最终融合：8路 softmax 加权融合（4视角 × 2特征 = 8路）
        merged = concatenate([base1, x1, base2, x2, base3, x3, base4, x4])
        outputs = Dense(self.classes, activation="softmax",
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(self.weight_decay))(merged)

        model_ = Model(inputs=[input1, input2, input3, input4], outputs=outputs)
        model_.summary()
        return model_

    def dynamic_weight_fusion_8(self, b1, x1, b2, x2, b3, x3, b4, x4):
        """8路 softmax 加权融合（4视角 × 2特征 = 8路）"""
        weights = layers.Concatenate(name='final_concat')([b1, x1, b2, x2, b3, x3, b4, x4])
        weights = layers.Dense(256, activation='relu', name='final_d1')(weights)
        weights = layers.Dense(8, activation='softmax', name='final_d2')(weights)
        w1 = layers.Multiply(name='final_b1')([b1, weights[:, 0:1]])
        w2 = layers.Multiply(name='final_x1')([x1, weights[:, 1:2]])
        w3 = layers.Multiply(name='final_b2')([b2, weights[:, 2:3]])
        w4 = layers.Multiply(name='final_x2')([x2, weights[:, 3:4]])
        w5 = layers.Multiply(name='final_b3')([b3, weights[:, 4:5]])
        w6 = layers.Multiply(name='final_x3')([x3, weights[:, 5:6]])
        w7 = layers.Multiply(name='final_b4')([b4, weights[:, 6:7]])
        w8 = layers.Multiply(name='final_x4')([x4, weights[:, 7:8]])
        return layers.Add(name='final_add')([w1, w2, w3, w4, w5, w6, w7, w8])
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
        # 数据路径（根据实际路径修改）
        train_dir = "D:\\数据集\\100-Driver\\splittrain"
        test_dir = "D:\\数据集\\100-Driver\\splittest"

        dir1_train = os.path.join(train_dir, "Cam1")
        dir2_train = os.path.join(train_dir, "Cam2")
        dir3_train = os.path.join(train_dir, "Cam3")
        dir4_train = os.path.join(train_dir, "Cam4")

        dir1_test = os.path.join(test_dir, "Cam1")
        dir2_test = os.path.join(test_dir, "Cam2")
        dir3_test = os.path.join(test_dir, "Cam3")
        dir4_test = os.path.join(test_dir, "Cam4")

        batch_size = self.batchsize
        sample = 2000

        # 训练集生成器
        train_generator = MultiDirectoryDataLoader(
            dir1=dir1_train, dir2=dir2_train, dir3=dir3_train, dir4=dir4_train,
            batch_size=batch_size, sampleNum=sample,
            img_size1=self.x_shape[:2], img_size2=self.x_shape[:2],
            img_size3=self.x_shape[:2], img_size4=self.x_shape[:2],
            shuffle=True, is_training=True
        )

        # 验证集生成器
        val_generator = MultiDirectoryDataLoader(
            dir1=dir1_test, dir2=dir2_test, dir3=dir3_test, dir4=dir4_test,
            batch_size=batch_size, sampleNum=sample,
            img_size1=self.x_shape[:2], img_size2=self.x_shape[:2],
            img_size3=self.x_shape[:2], img_size4=self.x_shape[:2],
            shuffle=False, is_training=False
        )

        num_classes = train_generator.get_num_classes()
        print(f"类别数: {num_classes}, 类别名: {train_generator.get_class_names()}")

        momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=0.0049, momentum=0.95)
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
