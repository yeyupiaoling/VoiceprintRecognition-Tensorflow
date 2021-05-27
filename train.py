import argparse
import functools
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from utils import reader
from utils.ArcMargin import ArcNet
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('num_epoch',        int,    200,                      '训练的轮数')
add_arg('num_classes',      int,    10,                     '分类的类别数量')
add_arg('learning_rate',    float,  1e-1,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(257, 257, 1)',          '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('use_model',        str,    'MobileNetV2',            '所使用的模型，支持MobileNetV2, ResNet50V2')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('pretrained_model', str,    None,                     '预训练模型的路径，当为None则不使用预训练模型')
add_arg('reg_weight_decay', float,  0.1,                      'weight decay for regression loss')
args = parser.parse_args()


# 评估模型
def test(model, metric_fc, test_dataset, loss_object, test_loss_metrics, test_accuracy_metrics):
    # 在下一个epoch开始时，重置评估指标
    test_loss_metrics.reset_states()
    test_accuracy_metrics.reset_states()
    # 开始评估
    for batch_data in test_dataset:
        sounds, labels = batch_data
        feature = model(sounds)
        predictions = metric_fc(feature, labels)
        # 获取损失值
        reg_loss = tf.reduce_sum(model.losses) * args.reg_weight_decay
        pred_loss = loss_object(labels, predictions)
        test_loss = pred_loss + reg_loss
        # 计算平均损失值和准确率
        test_loss_metrics(test_loss)
        test_accuracy_metrics(labels, predictions)
    return test_loss_metrics.result(), test_accuracy_metrics.result()


# 保存模型
def save_model(model, metric_fc):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    model.save(filepath=os.path.join(args.save_model, 'infer_model.h5'))
    model.save_weights(filepath=os.path.join(args.save_model, 'model_weights.h5'))
    metric_fc.save_weights(filepath=os.path.join(args.save_model, 'metric_fc_weights.h5'))


# 训练
def main():
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取模型
    model = tf.keras.Sequential()
    if args.use_model == 'MobileNetV2':
        model.add(MobileNetV2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
    else:
        model.add(ResNet50V2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(512, kernel_regularizer=tf.keras.regularizers.l2(5e-4), bias_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    metric_fc = ArcNet(feature_dim=512, n_classes=args.num_classes)

    # 打印模型
    model.build(input_shape=input_shape)
    model.summary()

    # 定义损失函数
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    with open(args.train_list_path, 'r') as f:
        lines = f.readlines()
    epoch_step_sum = int(len(lines) / args.batch_size)
    # 定义优化方法
    boundaries = [10 * epoch_step_sum, 30 * epoch_step_sum, 70 * epoch_step_sum, 100 * epoch_step_sum]
    lr = [0.5 ** l * args.learning_rate for l in range(len(boundaries) + 1)]
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

    # 获取训练和测试数据
    train_dataset = reader.train_reader(data_list_path=args.train_list_path,
                                        batch_size=args.batch_size,
                                        num_epoch=args.num_epoch,
                                        spec_len=input_shape[1])
    test_dataset = reader.test_reader(data_list_path=args.test_list_path,
                                      batch_size=args.batch_size,
                                      spec_len=input_shape[1])

    # 加载预训练模型
    if args.pretrained_model is not None:
        model.load_weights(os.path.join(args.save_model, 'model_weights.h5'))
        metric_fc(tf.convert_to_tensor(np.random.random((1, 512)), dtype='float32'), tf.convert_to_tensor([0]))
        metric_fc.load_weights(os.path.join(args.save_model, 'metric_fc_weights.h5'))
        print('加载预训练模型成功！')

    train_loss_metrics = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss_metrics = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_summary_writer = tf.summary.create_file_writer('log/train')
    test_summary_writer = tf.summary.create_file_writer('log/test')

    # 开始训练
    train_loss_metrics.reset_states()
    train_accuracy_metrics.reset_states()
    count_step = epoch_step_sum * args.num_epoch
    test_step = 0
    # 开始训练
    for step, batch_data in enumerate(train_dataset):
        sounds, labels = batch_data
        # 执行训练
        with tf.GradientTape() as tape:
            feature = model(sounds)
            predictions = metric_fc(feature, labels)
            # 获取损失值
            model_reg_loss = tf.reduce_sum(model.losses) * args.reg_weight_decay
            metric_fc_reg_loss = tf.reduce_sum(metric_fc.losses) * args.reg_weight_decay
            pred_loss = loss_object(labels, predictions)
            train_loss = pred_loss + model_reg_loss + metric_fc_reg_loss

        # 更新梯度
        trainable_variables = model.trainable_variables + metric_fc.trainable_variables
        gradients = tape.gradient(train_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # 计算平均损失值和准确率
        train_loss_metrics(train_loss)
        train_accuracy_metrics(labels, predictions)
        # 日志输出
        if step % 10 == 0:
            print("[%s] Step [%d/%d], Loss %f, Accuracy %f" % (
                datetime.now(), step, count_step, train_loss_metrics.result(), train_accuracy_metrics.result()))

        # 记录数据
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', train_loss_metrics.result(), step=step)
            tf.summary.scalar('Accuracy', train_accuracy_metrics.result(), step=step)

        # 评估模型
        if step % epoch_step_sum == 0 and step != 0:
            test_loss, test_accuracy = test(model, metric_fc, test_dataset, loss_object, test_loss_metrics, test_accuracy_metrics)
            print('=================================================')
            print("[%s] Test Loss %f, Accuracy %f" % (datetime.now(), test_loss, test_accuracy))
            print('=================================================')
            # 记录数据
            with test_summary_writer.as_default():
                tf.summary.scalar('Loss', test_loss, step=test_step)
                tf.summary.scalar('Accuracy', test_accuracy, step=test_step)
            test_step += 1

            # 保存模型
            save_model(model, metric_fc)


if __name__ == '__main__':
    print_arguments(args)
    main()
