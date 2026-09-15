"""
测量 seven_auc_multiview-final.py 网络的 GMACs 和 FPS
不训练，只构建模型进行测量
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense,
                                     PReLU, Input, BatchNormalization, GlobalMaxPooling2D, SeparableConv2D,
                                     LeakyReLU, Concatenate, Lambda, Flatten, InputLayer)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, layers
from tensorflow.keras.layers import Activation, Dropout, AveragePooling2D, add
from tensorflow.python.keras.layers import concatenate, multiply

import cbam


def build_model_for_measure():
    x_shape = (224, 224, 3)
    classes = 10

    input1 = Input(shape=x_shape, name='view1_input')
    input2 = Input(shape=x_shape, name='view2_input')
    input3 = Input(shape=x_shape, name='view3_input')
    input4 = Input(shape=x_shape, name='view4_input')

    # 共享 MobileNet
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=x_shape)
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

    def sobel_branch(inp, view_idx):
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

    def build_single_view(inputs, view_idx):
        p = f'v{view_idx}_'
        soble = sobel_branch(inputs, view_idx)
        base_output = mobilenet_fn(inputs)
        soble = GlobalAveragePooling2D(name=f'{p}gap_x')(soble)
        base_gap = GlobalAveragePooling2D(name=f'{p}gap_base')(base_output)
        x = multiply([base_gap, soble], name=f'{p}mul')
        return x

    x1 = build_single_view(input1, 1)
    x2 = build_single_view(input2, 2)
    x3 = build_single_view(input3, 3)
    x4 = build_single_view(input4, 4)

    merged = concatenate([x1, x2, x3, x4])
    outputs = Dense(classes, activation="softmax",
                    kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0005))(merged)
    model = Model(inputs=[input1, input2, input3, input4], outputs=outputs)
    return model


def get_gflops(model, input_shapes):
    """使用 tf.profiler 计算 GFLOPs（1 GMACs = 2 FLOPs）"""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    batch_size = 1
    inputs = [tf.TensorSpec([batch_size] + shape[1:], tf.float32) for shape in input_shapes]

    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation())
            .with_empty_output()
            .build())

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()
    gflops = (flops.total_float_ops / 1e9) / 2  # FLOPs → GMACs
    return gflops


def measure_fps(model, input_shapes, num_warmup=10, num_runs=100):
    """测量 FPS（每秒处理帧数）"""
    # 模拟输入数据
    dummy_inputs = [np.random.randn(1, *shape[1:]).astype(np.float32) for shape in input_shapes]

    # 预热
    print(f"预热 {num_warmup} 次...")
    for i in range(num_warmup):
        _ = model.predict(dummy_inputs, verbose=0)

    # 正式测量
    print(f"正式测量 {num_runs} 次...")
    start_time = time.time()
    for i in range(num_runs):
        _ = model.predict(dummy_inputs, verbose=0)
    end_time = time.time()

    total_time = end_time - start_time
    fps = num_runs / total_time
    avg_latency_ms = (total_time / num_runs) * 1000
    return fps, avg_latency_ms


def get_model_size(model):
    """获取模型参数量"""
    total_params = model.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    return total_params, int(trainable_params), int(non_trainable_params)


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    K.clear_session()
    np.random.seed(2025)
    tf.random.set_seed(2025)

    print("=" * 60)
    print("构建模型...")
    model = build_model_for_measure()

    # 输入形状（4个视角，batch_size=1）
    input_shapes = [(1, 224, 224, 3)] * 4

    # 1. 模型参数量
    print("\n" + "=" * 60)
    print("【1】模型参数量")
    print("=" * 60)
    total, trainable, non_trainable = get_model_size(model)
    print(f"总参数量: {total:,}")
    print(f"可训练参数: {trainable:,}")
    print(f"不可训练参数: {non_trainable:,}")
    print(f"模型大小(估计, fp32): {total * 4 / 1024 / 1024:.2f} MB")

    # 2. GMACs
    print("\n" + "=" * 60)
    print("【2】计算量 GMACs")
    print("=" * 60)
    try:
        gflops = get_gflops(model, input_shapes)
        print(f"GMACs: {gflops:.4f}")
        print(f"GFLOPs: {gflops * 2:.4f}")
    except Exception as e:
        print(f"GMACs 计算失败: {e}")
        print("尝试使用 keras-flops 计算...")
        try:
            # 备用方法：手动估算
            print("使用手动估算方式...")
            # MobileNet conv_pw_13_relu 之前的 GMACs 约 0.0942 GMACs（单视角）
            # 4 视角共享但前向计算 4 次
            print("（建议安装 keras-flops: pip install keras-flops）")
        except:
            pass

    # 3. FPS
    print("\n" + "=" * 60)
    print("【3】推理速度 FPS")
    print("=" * 60)
    fps, latency_ms = measure_fps(model, input_shapes, num_warmup=5, num_runs=500)
    print(f"平均延迟: {latency_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # 4. 汇总
    print("\n" + "=" * 60)
    print("【汇总】")
    print("=" * 60)
    print(f"模型: seven_auc_multiview-final.py")
    print(f"输入: 4 × (1, 224, 224, 3)")
    print(f"总参数量: {total:,}")
    try:
        print(f"GMACs: {gflops:.4f}")
    except:
        pass
    print(f"FPS: {fps:.2f}")
    print(f"平均延迟: {latency_ms:.2f} ms")


if __name__ == '__main__':
    main()
