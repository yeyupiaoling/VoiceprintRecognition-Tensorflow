import tensorflow as tf
import reader
import numpy as np

class_dim = 855
EPOCHS = 500
BATCH_SIZE=32

model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(128, None, 1)),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
])

model.summary()

# 定义优化方法
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_dataset = reader.train_reader_tfrecord('dataset/train.tfrecord', EPOCHS, batch_size=BATCH_SIZE)
test_dataset = reader.test_reader_tfrecord('dataset/test.tfrecord', batch_size=BATCH_SIZE)

for batch_id, data in enumerate(train_dataset):
    # [可能需要修改参数】 设置的梅尔频谱的shape
    sounds = data['data'].numpy().reshape((BATCH_SIZE, 128, 128, 1))
    labels = data['label']
    # 执行训练
    with tf.GradientTape() as tape:
        predictions = model(sounds)
        # 获取损失值
        train_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        train_loss = tf.reduce_mean(train_loss)
        # 获取准确率
        train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, predictions)
        train_accuracy = np.sum(train_accuracy.numpy()) / len(train_accuracy.numpy())

    # 更新梯度
    gradients = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if batch_id % 20 == 0:
        print("Batch %d, Loss %f, Accuracy %f" % (batch_id, train_loss.numpy(), train_accuracy))

    if batch_id % 200 == 0 and batch_id != 0:
        test_losses = list()
        test_accuracies = list()
        for d in test_dataset:
            # [可能需要修改参数】 设置的梅尔频谱的shape
            test_sounds = d['data'].numpy().reshape((-1, 128, 128, 1))
            test_labels = d['label']

            test_result = model(test_sounds)
            # 获取损失值
            test_loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, test_result)
            test_loss = tf.reduce_mean(test_loss)
            test_losses.append(test_loss)
            # 获取准确率
            test_accuracy = tf.keras.metrics.sparse_categorical_accuracy(test_labels, test_result)
            test_accuracy = np.sum(test_accuracy.numpy()) / len(test_accuracy.numpy())
            test_accuracies.append(test_accuracy)

        print('=================================================')
        print("Test, Loss %f, Accuracy %f" % (
            sum(test_losses) / len(test_losses), sum(test_accuracies) / len(test_accuracies)))
        print('=================================================')

        # 保存模型
        model.save(filepath='models/resnet.h5')
