
import os
import random

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import math

class MultiDirectoryDataLoader(Sequence):
    val=None

    valData=None
    """
    多目录数据加载器，支持从不同目录加载不同分支的图像数据
    """
    def __init__(self, dir1, dir2,dir3,dir4, batch_size=16, sampleNum=300, img_size1=(224, 224), img_size2=(224, 224), img_size3=(224, 224), img_size4=(224, 224),
                 shuffle=True, validation_split=0.2, is_training=True):
        """
        初始化多目录数据加载器
        Args:
            dir1: 第一个分支图像目录
            dir2: 第二个分支图像目录
            batch_size: 批次大小
            img_size1: 第一个分支图像尺寸
            img_size2: 第二个分支图像尺寸
            shuffle: 是否打乱数据
            validation_split: 验证集比例
            is_training: 是否为训练集
        """
        self.dir1 = dir1
        self.dir2 = dir2
        self.dir3 = dir3
        self.dir4 = dir4
        self.batch_size = batch_size
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        self.img_size3 = img_size3
        self.img_size4 = img_size4
        self.shuffle = shuffle
        
        # 收集所有图像路径和标签
        self.image_paths1 = []
        self.image_paths11 = []
        self.image_paths2 = []
        self.image_paths21 = []
        self.image_paths3 = []
        self.image_paths31 = []
        self.image_paths4 = []
        self.image_paths41 = []
        self.labels = []
        self.labels1 = []
        self.classes = []

        self.sampleNum=sampleNum  #定义每个类别的样本数量

        self.val_split = []

        
        # 获取所有类别
        classes1 = set(os.listdir(dir1))
        classes2 = set(os.listdir(dir2))
        classes3 = set(os.listdir(dir3))
        classes4 = set(os.listdir(dir4))
        self.classes = sorted(list(classes1.intersection(classes2,classes3,classes4)))
        
        if not self.classes:
            raise ValueError("两个目录中没有共同的类别文件夹")
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        # 收集图像路径
        for class_name in self.classes:
            class_dir1 = os.path.join(dir1, class_name)
            class_dir2 = os.path.join(dir2, class_name)
            class_dir3 = os.path.join(dir3, class_name)
            class_dir4 = os.path.join(dir4, class_name)

            if not os.path.isdir(class_dir1) or not os.path.isdir(class_dir2) or not os.path.isdir(class_dir3) or not os.path.isdir(class_dir4):
                continue

            # 随机抽取数据集
            image_files1 = [f for f in os.listdir(class_dir1)[:self.sampleNum] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files2 = [f for f in os.listdir(class_dir2)[:self.sampleNum] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files3 = [f for f in os.listdir(class_dir3)[:self.sampleNum] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files4 = [f for f in os.listdir(class_dir4)[:self.sampleNum] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            np.random.seed(2025)
            np.random.shuffle(image_files1)
            np.random.shuffle(image_files2)
            np.random.shuffle(image_files3)
            np.random.shuffle(image_files4)

            for img_file in image_files1:
                self.image_paths1.append(os.path.join(class_dir1, img_file))

            for img_file in image_files2:
                self.image_paths2.append(os.path.join(class_dir2, img_file))

            for img_file in image_files3:
                self.image_paths3.append(os.path.join(class_dir3, img_file))

            #最后一轮收集标签
            for img_file in image_files4:
                self.image_paths4.append(os.path.join(class_dir4, img_file))
                self.labels.append(class_name)

        self.num_classes = len(self.classes)
        self.indexes = np.arange(len(self.image_paths1))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """返回批次数"""
        return math.ceil(self.sampleNum / self.batch_size)
    
    def __getitem__(self, index):
        """生成一个批次的数据"""
        # 生成索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # 找到批次对应的图像路径和标签
        batch_paths1 = [self.image_paths1[k] for k in indexes]
        batch_paths2 = [self.image_paths2[k] for k in indexes]
        batch_paths3 = [self.image_paths3[k] for k in indexes]
        batch_paths4 = [self.image_paths4[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        
        # 生成数据
        X, y = self._data_generation(batch_paths1, batch_paths2,batch_paths3,batch_paths4, batch_labels)
        
        return X, y
    
    def _data_generation(self, batch_paths1, batch_paths2,batch_paths3,batch_paths4, batch_labels):
        """生成批次数据"""
        # 初始化批次数据
        batch_size = len(batch_paths1)
        images1 = np.empty((batch_size, *self.img_size1, 3))
        images2 = np.empty((batch_size, *self.img_size2, 3))
        images3 = np.empty((batch_size, *self.img_size3, 3))
        images4 = np.empty((batch_size, *self.img_size4, 3))
        labels = np.empty((batch_size,), dtype=int)
        
        # 生成数据
        for i, (img_path1, img_path2,img_path3,img_path4, label) in enumerate(zip(batch_paths1, batch_paths2,batch_paths3,batch_paths4, batch_labels)):
            # 加载第一个分支的图像 (RGB)
            img1 = load_img(img_path1, target_size=self.img_size1)
            images1[i,] = img_to_array(img1)
            
            # 加载第二个分支的图像 (RGB)
            img2 = load_img(img_path2, target_size=self.img_size2)
            images2[i,] = img_to_array(img2)

            # 加载第二个分支的图像 (RGB)
            img3 = load_img(img_path3, target_size=self.img_size3)
            images3[i,] = img_to_array(img3)

            # 加载第二个分支的图像 (RGB)
            img4 = load_img(img_path4, target_size=self.img_size4)
            images4[i,] = img_to_array(img4)
            
            # 存储类别
            labels[i] = self.label_encoder.transform([label])[0]
        
        # 转换标签为one-hot编码
        labels = np.eye(self.num_classes)[labels]
        return [images1, images2,images3,images4], labels
    
    def on_epoch_end(self):
        """每个epoch结束后调用"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes
    
    def get_class_names(self):
        """获取类别名称"""
        return self.classes


class SingleDirectoryDataLoader(Sequence):
    """
    单目录数据加载器：只从一个目录加载单一视角的图像数据
    用于单视角训练和测试（例如只训练 Cam4）
    """
    def __init__(self, data_dir, batch_size=16, sampleNum=300, img_size=(224, 224),
                 shuffle=True, is_training=True):
        """
        初始化单目录数据加载器
        Args:
            data_dir: 图像目录（包含按类别划分的子目录）
            batch_size: 批次大小
            sampleNum: 每个类别的样本数量上限
            img_size: 图像尺寸
            shuffle: 是否打乱数据
            is_training: 是否为训练集
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.sampleNum = sampleNum

        self.image_paths = []
        self.labels = []
        self.classes = []

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])

        if not self.classes:
            raise ValueError(f"目录 {data_dir} 中没有类别文件夹")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        # 收集图像路径
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir)[:self.sampleNum]
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            np.random.seed(2025)
            np.random.shuffle(image_files)

            for img_file in image_files:
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.labels.append(class_name)

        self.num_classes = len(self.classes)
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """返回批次数"""
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        """生成一个批次的数据"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]
        X, y = self._data_generation(batch_paths, batch_labels)
        return X, y

    def _data_generation(self, batch_paths, batch_labels):
        """生成批次数据"""
        batch_size = len(batch_paths)
        images = np.empty((batch_size, *self.img_size, 3))
        labels = np.empty((batch_size,), dtype=int)

        for i, (img_path, label) in enumerate(zip(batch_paths, batch_labels)):
            img = load_img(img_path, target_size=self.img_size)
            images[i,] = img_to_array(img)
            labels[i] = self.label_encoder.transform([label])[0]

        labels = np.eye(self.num_classes)[labels]
        return images, labels

    def on_epoch_end(self):
        """每个epoch结束后调用"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes

    def get_class_names(self):
        """获取类别名称"""
        return self.classes
