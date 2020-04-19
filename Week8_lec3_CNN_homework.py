# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:09:37 2020
y = <W, X>  dy/dw = X(X.T)

一共两份大作业:  第一份CNN的 numpy 版， 还有一份是算法部分，  Adam, RMSprop
@author: Lloyd
"""
import tensorflow as tf
input_shape = (4, 28, 28, 3)
# 四个样本， size: 28x28,  channel: 3 
# 四幅彩色图像
# strides是一项技术。
# science and technology区别
# science: 原理
# 技术: 怎么好就怎么用。 CNN最常用的是识别，只要提取的特征能帮助识别就行了
# 后面会涉及到高分辨率图像， 比如ImageNet
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
    2, 3, strides=(2, 3),  activation='relu', input_shape=input_shape)(x)
print(y)

