
# 前言
本章介绍如何使用Tensorflow实现简单的声纹识别模型，首先你需要熟悉音频分类，没有了解的可以查看这篇文章[《基于Tensorflow实现声音分类》](https://blog.doiduoyi.com/articles/1587654005620.html)。基于这个知识基础之上，我们训练一个声纹识别模型，通过这个模型我们可以识别说话的人是谁，可以应用在一些需要音频验证的项目。


# 环境准备
主要介绍libsora，PyAudio，pydub的安装，其他的依赖包根据需要自行安装。
 - Python 3.7
 - Tensorflow 2.0

## 安装libsora
最简单的方式就是使用pip命令安装，如下：
```shell
pip install pytest-runner
pip install librosa
```

如果pip命令安装不成功，那就使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/)， windows的可以下载zip压缩包，方便解压。
```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现`libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如`pip install librosa==0.6.3`

## 安装PyAudio
使用pip安装命令，如下：
```shell
pip install pyaudio
```
 在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

## 安装pydub
使用pip命令安装，如下：
```shell
pip install pydub
```


# 创建数据
本教程笔者使用的是[Free ST Chinese Mandarin Corpus数据集](http://www.openslr.org/38)，这个数据集一共有855个人的语音数据，有102600条语音数据。如果读者有其他更好的数据集，可以混合在一起使用。

如何已经读过笔者[《基于Tensorflow实现声音分类》](https://blog.doiduoyi.com/articles/1587654005620.html)这篇文章，应该知道语音数据小而多，最好的方法就是把这些音频文件生成TFRecord，加快训练速度。所以创建`create_data.py`用于生成TFRecord文件。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中，这样就可以在下一步直接生成TFRecord文件了。
```python
def get_data_list(audio_path, list_path):
    files = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    sound_sum = 0
    s = set()
    for file in files:
        if '.wav' not in file:
            continue
        s.add(file[:15])
        sound_path = os.path.join(audio_path, file)
        if sound_sum % 100 == 0:
            f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), len(s) - 1))
        else:
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), len(s) - 1))
        sound_sum += 1

    f_test.close()
    f_train.close()

if __name__ == '__main__':
    get_data_list('dataset/ST-CMDS-20170001_1-OS', 'dataset')
```


有了上面创建的数据列表，就可以把语音数据转换成训练数据了，主要是把语音数据转换成梅尔频谱（Mel Spectrogram），使用librosa可以很方便得到音频的梅尔频谱，使用的API为`librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用tensorflow训练和预测。关于梅尔频谱具体信息读者可以自行了解，跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为`librosa.feature.mfcc()`。在转换过程中，笔者还使用了`librosa.effects.split`裁剪掉静音部分的音频，这样可以减少训练数据的噪声，提供训练准确率。笔者目前默认每条语音的长度为2.04秒，这个读者可以根据自己的情况修改语音的长度，如果要修改训练语音的长度，需要根据注释的提示修改相应的数据值。如果语音长度比较长的，程序会随机裁剪20次，以达到数据增强的效果。
```python
# 获取浮点数组
def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# 获取整型数据
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 把数据添加到TFRecord中
def data_example(data, label):
    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# 开始创建tfrecord数据
def create_data_tfrecord(data_list_path, save_path):
    with open(data_list_path, 'r') as f:
        data = f.readlines()
    with tf.io.TFRecordWriter(save_path) as writer:
        for d in tqdm(data):
            try:
                path, label = d.replace('\n', '').split('\t')
                wav, sr = librosa.load(path, sr=16000)
                intervals = librosa.effects.split(wav, top_db=20)
                wav_output = []
                # [可能需要修改参数] 音频长度 16000 * 秒数
                wav_len = int(16000 * 2.04)
                for sliced in intervals:
                    wav_output.extend(wav[sliced[0]:sliced[1]])
                for i in range(20):
                    # 裁剪过长的音频，过短的补0
                    if len(wav_output) > wav_len:
                        l = len(wav_output) - wav_len
                        r = random.randint(0, l)
                        wav_output = wav_output[r:wav_len + r]
                    else:
                        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
                    wav_output = np.array(wav_output)
                    # 转成梅尔频谱
                    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).reshape(-1).tolist()
                    # [可能需要修改参数] 梅尔频谱shape ，librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).shape
                    if len(ps) != 128 * 128: continue
                    tf_example = data_example(ps, int(label))
                    writer.write(tf_example.SerializeToString())
                    if len(wav_output) <= wav_len:
                        break
            except Exception as e:
                print(e)
if __name__ == '__main__':
    create_data_tfrecord('dataset/train_list.txt', 'dataset/train.tfrecord')
    create_data_tfrecord('dataset/test_list.txt', 'dataset/test.tfrecord')
```

在上面已经创建了TFRecord文件，为了可以在训练中读取TFRecord文件，创建`reader.py`程序用于读取训练数据，如果读者已经修改了训练数据的长度，需要修改`tf.io.FixedLenFeature`中的值。
```python
def _parse_data_function(example):
    # [可能需要修改参数】 设置的梅尔频谱的shape相乘的值
    data_feature_description = {
        'data': tf.io.FixedLenFeature([16384], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, data_feature_description)


def train_reader_tfrecord(data_path, num_epochs, batch_size):
    raw_dataset = tf.data.TFRecordDataset(data_path)
    train_dataset = raw_dataset.map(_parse_data_function)
    train_dataset = train_dataset.shuffle(buffer_size=1000) \
        .repeat(count=num_epochs) \
        .batch(batch_size=batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset


def test_reader_tfrecord(data_path, batch_size):
    raw_dataset = tf.data.TFRecordDataset(data_path)
    test_dataset = raw_dataset.map(_parse_data_function)
    test_dataset = test_dataset.batch(batch_size=batch_size)
    return test_dataset
```


# 训练模型
创建`train.py`开始训练模型，搭建一个ResNet50分类模型，`input_shape`设置为`(128, None, 1))`主要是为了适配其他音频长度的输入和预测是任意大小的输入。`class_dim`为分类的总数，Free ST Chinese Mandarin Corpus数据集一共有855个人的语音数据，所以这里分类总数为855，可以使用之前训练过的权重初始化模型，下载看文章最后。
```python
class_dim = 855
EPOCHS = 500
BATCH_SIZE=32
init_model = "models/model_weights.h5"

model = tf.keras.models.Sequential([
    tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=(128, None, 1)),
    tf.keras.layers.ActivityRegularization(l2=0.5),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(units=class_dim, activation=tf.nn.softmax)
])

model.summary()

# 定义优化方法
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_dataset = reader.train_reader_tfrecord('dataset/train.tfrecord', EPOCHS, batch_size=BATCH_SIZE)
test_dataset = reader.test_reader_tfrecord('dataset/test.tfrecord', batch_size=BATCH_SIZE)

if init_model:
    model.load_weights(init_model)
```

开始执行训练，要注意的是在创建TFRecord文件时，已经把音频数据的梅尔频谱转换为一维list了，所以在数据输入到模型前，需要把数据reshape为之前的shape，操作方式为`reshape((-1, 128, 128, 1))`。要注意的是如果读者使用了其他长度的音频，需要根据梅尔频谱的shape修改，训练数据和测试数据都需要做同样的处理。每训练200个batch执行一次测试和保存模型，包括预测模型和网络权重。
```python
for batch_id, data in enumerate(train_dataset):
    # [可能需要修改参数】 设置的梅尔频谱的shape
    sounds = data['data'].numpy().reshape((-1, 128, 128, 1))
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
        model.save_weights(filepath='models/model_weights.h5')
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，在加载模型时，不要直接加载整个模型，而是加载模型的最后分类层的上一层，这样就可以获取到语音的特征数据。通过使用[netron](https://github.com/lutzroeder/netron)查看每一层的输入和输出的名称。

```python
layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
```

然后编写两个函数，分类是加载数据和执行预测的函数，在这个加载数据函数中并没有限定输入音频的大小，只是不允许裁剪静音后的音频不能小于0.5秒，这样就可以输入任意长度的音频。执行预测之后数据的是语音的特征值。
```python
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    assert len(wav_output) >= 8000, "有效音频小于0.5s"
    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature
```

有了上面两个函数，就可以做声纹识别了。我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值，读者可以根据自己项目的准确度要求进行修改。
```python
if __name__ == '__main__':
    # 要预测的两个人的音频文件
    person1 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00011A0001.wav'
    person2 = 'dataset/ST-CMDS-20170001_1-OS/20170001P00011I0081.wav'
    feature1 = infer(person1)[0]
    feature2 = infer(person2)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.7:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
```

# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的数据加载函数和预测函数，通过这两个同样获取语音的特征数据。
```python
layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

person_feature = []
person_name = []


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    if len(wav_output) < 8000:
        raise Exception("有效音频小于0.5s")
    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature
```

不同的是笔者增加了`load_audio_db()`和`recognition()`，第一个函数是加载语音库中的语音数据，这些音频就是相当于已经注册的用户，他们注册的语音数据会存放在这里，如果有用户需要通过声纹登录，就需要拿到用户的语音和语音库中的语音进行声纹对比，如果对比成功，那就相当于登录成功并且获取用户注册时的信息数据。完成识别的主要在`recognition()`函数中，这个函数就是将输入的语音和语音库中的语音一一对比。
```python
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro
```

有了上面的声纹识别的函数，读者可以根据自己项目的需求完成声纹识别的方式，例如笔者下面提供的是通过录音来完成声纹识别。首先必须要加载语音库中的语音，语音库文件夹为`audio_db`，然后用户回车后录音3秒钟，然后程序会自动录音，并使用录音到的音频进行声纹识别，去匹配语音库中的语音，获取用户的信息。通过这样方式，读者也可以修改成通过服务请求的方式完成声纹识别，例如提供一个API供APP调用，用户在APP上通过声纹登录时，把录音到的语音发送到后端完成声纹识别，再把结果返回给APP，前提是用户已经使用语音注册，并成功把语音数据存放在`audio_db`文件夹中。
```python
if __name__ == '__main__':
    load_audio_db('audio_db')
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"

    # 打开录音
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while True:
        try:
            i = input("按下回车键开机录音，录音3秒中：")
            print("开始录音......")
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("录音已结束!")

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # 识别对比音频库的音频
            name, p = recognition(WAVE_OUTPUT_FILENAME)
            if p > 0.7:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        except:
            pass
```

# 模型

| 模型名称 | 所用数据集 | 下载地址 |
| :---: | :---: | :---: |
| 网络权重 | ST-CMDS-20170001_1-OS | [点击下载](https://resource.doiduoyi.com/#q81u0uw) |
| 网络预测模型 | ST-CMDS-20170001_1-OS | [点击下载](https://resource.doiduoyi.com/#4g6u00i) |
| 网络预测模型 | 更大数据集 | |
