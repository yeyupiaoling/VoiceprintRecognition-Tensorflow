import argparse
import functools
import os
import shutil
import time
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense

from utils import reader
from utils.loss import ArcLoss
from utils.metrics import ArcNet
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,    '0',                      '训练使用的GPU序号，使用英文逗号,隔开，如：0,1')
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('num_classes',      int,    3242,                     '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(257, 257, 1)',          '数据输入的形状')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model_path',  str,    'models/',                '模型保存的路径')
add_arg('pretrained_model', str,    'models/model_weights.h5','预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()


# 保存模型
def save_model(model):
    os.makedirs(args.save_model_path, exist_ok=True)
    infer_model = Model(inputs=model.input, outputs=model.get_layer('feature_output').output)
    infer_model.save(filepath=os.path.join(args.save_model_path, 'infer_model.h5'), include_optimizer=False)
    model.save_weights(filepath=os.path.join(args.save_model_path, 'model_weights.h5'))
    print('模型保存成功！')


def create_model(input_shape):
    # 获取模型
    model = tf.keras.Sequential()
    model.add(ResNet50V2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
    model.add(BatchNormalization())
    model.add(Dense(units=512, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='feature_output'))
    model.add(ArcNet(num_classes=args.num_classes))
    return model


# 训练
def main():
    shutil.rmtree('log', ignore_errors=True)
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 支持多卡训练
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with open(args.train_list_path, 'r') as f:
        lines = f.readlines()
    epoch_step_sum = int(len(lines) / BATCH_SIZE)

    # 获取训练和测试数据
    train_dataset = reader.train_reader(data_list_path=args.train_list_path,
                                        batch_size=BATCH_SIZE,
                                        num_epoch=args.num_epoch,
                                        spec_len=input_shape[1])
    test_dataset = reader.test_reader(data_list_path=args.test_list_path,
                                      batch_size=BATCH_SIZE,
                                      spec_len=input_shape[1])
    # 支持多卡训练
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    with strategy.scope():
        model = create_model(input_shape)
        # 打印模型
        model.build(input_shape=input_shape)
        model.summary()
        # 定义优化方法
        boundaries = [10 * i * epoch_step_sum for i in range(1, args.num_epoch // 10, 1)]
        lr = [0.1 ** l * args.learning_rate for l in range(len(boundaries) + 1)]
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9)

    with strategy.scope():
        # 加载预训练模型
        if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
            model.load_weights(args.pretrained_model, by_name=True, skip_mismatch=True)
            print('加载预训练模型成功！')

    with strategy.scope():
        train_loss_metrics = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss_metrics = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        train_loss_metrics.reset_states()
        train_accuracy_metrics.reset_states()

        train_summary_writer = tf.summary.create_file_writer('log/train')
        test_summary_writer = tf.summary.create_file_writer('log/test')

    with strategy.scope():
        # 定义损失函数
        loss_object = ArcLoss(num_classes=args.num_classes, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(label, prediction):
            per_example_loss = loss_object(prediction, label)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    with strategy.scope():
        def train_step(inputs):
            sounds, labels = inputs
            # 执行训练
            with tf.GradientTape() as tape:
                predictions = model(sounds)
                # 获取损失值
                train_loss = compute_loss(labels, predictions)

            # 更新梯度
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # 计算平均损失值和准确率
            train_loss_metrics(train_loss)
            train_accuracy_metrics(labels, predictions)
            return train_loss

        def test_step(inputs):
            sounds, labels = inputs
            # 开始评估
            predictions = model(sounds)
            # 获取损失值
            test_loss = compute_loss(labels, predictions)
            # 计算平均损失值和准确率
            test_loss_metrics(test_loss)
            test_accuracy_metrics(labels, predictions)

    with strategy.scope():
        # `run`将复制提供的计算并使用分布式输入运行它。
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

        # 开始训练
        train_step_num = 0
        test_step_num = 0
        count_step = epoch_step_sum * args.num_epoch
        start = time.time()
        for step, train_inputs in enumerate(train_dataset):
            distributed_train_step(train_inputs)

            # 日志输出
            if step % 100 == 0:
                eta_sec = ((time.time() - start) * 1000) * (count_step - step)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print("[%s] Step [%d/%d], Loss %f, Accuracy %f, Learning rate %f, eta: %s" % (
                    datetime.now(), step, count_step, train_loss_metrics.result(), train_accuracy_metrics.result(),
                    optimizer.learning_rate.numpy(), eta_str))

                # 记录数据
                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', train_loss_metrics.result(), step=train_step_num)
                    tf.summary.scalar('Accuracy', train_accuracy_metrics.result(), step=train_step_num)
                train_step_num += 1
            # 评估模型
            if step % epoch_step_sum == 0 and step != 0:
                for test_inputs in test_dataset:
                    distributed_test_step(test_inputs)
                print('=================================================')
                print("[%s] Test Loss %f, Accuracy %f" % (datetime.now(), test_loss_metrics.result(), test_accuracy_metrics.result()))
                print('=================================================')
                # 记录数据
                with test_summary_writer.as_default():
                    tf.summary.scalar('Loss', test_loss_metrics.result(), step=test_step_num)
                    tf.summary.scalar('Accuracy', test_accuracy_metrics.result(), step=test_step_num)
                test_step_num += 1
                test_loss_metrics.reset_states()
                test_accuracy_metrics.reset_states()

                # 保存模型
                save_model(model)
            start = time.time()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print_arguments(args)
    main()
