import argparse
import functools
import os
import shutil
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from utils import reader
from utils.loss import ArcLoss
from utils.metrics import ArcNet
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    3242,                     '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(257, 257, 1)',          '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('pretrained_model', str,    None,                     '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 评估模型
def test(model, test_dataset, loss_object, test_loss_metrics, test_accuracy_metrics):
    # 在下一个epoch开始时，重置评估指标
    test_loss_metrics.reset_states()
    test_accuracy_metrics.reset_states()
    # 开始评估
    for sounds, labels in test_dataset:
        predictions = model(sounds)
        # 获取损失值
        test_loss = loss_object(predictions, labels)
        # 计算平均损失值和准确率
        test_loss_metrics(test_loss)
        test_accuracy_metrics(labels, predictions)
    return test_loss_metrics.result(), test_accuracy_metrics.result()


# 保存模型
def save_model(model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    model.save(filepath=os.path.join(args.save_model, 'infer_model.h5'))
    model.save_weights(filepath=os.path.join(args.save_model, 'model_weights.h5'))


# 训练
def main():
    shutil.rmtree('log', ignore_errors=True)
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取模型
    model = tf.keras.Sequential()
    model.add(ResNet50V2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(512, kernel_regularizer=tf.keras.regularizers.l2(5e-4), bias_initializer='glorot_uniform'))
    model.add(BatchNormalization(name='feature_output'))
    model.add(ArcNet(num_classes=args.num_classes))

    # 打印模型
    model.build(input_shape=input_shape)
    model.summary()

    with open(args.train_list_path, 'r') as f:
        lines = f.readlines()
    epoch_step_sum = int(len(lines) / args.batch_size)
    # 定义优化方法
    boundaries = [10 * i * epoch_step_sum for i in range(1, args.num_epoch // 10, 1)]
    lr = [0.1 ** l * args.learning_rate for l in range(len(boundaries) + 1)]
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=lr)
    optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9)

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
        print('加载预训练模型成功！')

    train_loss_metrics = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss_metrics = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_summary_writer = tf.summary.create_file_writer('log/train')
    test_summary_writer = tf.summary.create_file_writer('log/test')

    # 定义损失函数
    loss_object = ArcLoss(num_classes=args.num_classes)
    # 开始训练
    train_loss_metrics.reset_states()
    train_accuracy_metrics.reset_states()
    count_step = epoch_step_sum * args.num_epoch
    test_step = 0
    # 开始训练
    for step, (sounds, labels) in enumerate(train_dataset):
        # 执行训练
        with tf.GradientTape() as tape:
            predictions = model(sounds)
            # 获取损失值
            train_loss = loss_object(predictions, labels)

        # 更新梯度
        gradients = tape.gradient(train_loss, model.trainable_variables)
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
            test_loss, test_accuracy = test(model, test_dataset, loss_object, test_loss_metrics, test_accuracy_metrics)
            print('=================================================')
            print("[%s] Test Loss %f, Accuracy %f" % (datetime.now(), test_loss, test_accuracy))
            print('=================================================')
            # 记录数据
            with test_summary_writer.as_default():
                tf.summary.scalar('Loss', test_loss, step=test_step)
                tf.summary.scalar('Accuracy', test_accuracy, step=test_step)
            test_step += 1

            # 保存模型
            save_model(model)


if __name__ == '__main__':
    print_arguments(args)
    main()
