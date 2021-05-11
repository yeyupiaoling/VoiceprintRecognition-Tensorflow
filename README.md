
# 前言
本章介绍如何使用Tensorflow实现简单的声纹识别模型，首先你需要熟悉音频分类，没有了解的可以查看这篇文章[《基于Tensorflow实现声音分类》](https://blog.doiduoyi.com/articles/1587654005620.html)。基于这个知识基础之上，我们训练一个声纹识别模型，通过这个模型我们可以识别说话的人是谁，可以应用在一些需要音频验证的项目。


# 模型下载
| 数据集 | 准确率 | 下载地址 |
| :---: | :---: | :---: |
| [中文语音语料数据集](https://github.com/KuangDD/zhvoice) | 训练中 | [训练中]() |

# 安装环境
最简单的方式就是使用pip命令安装，如下：
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

注意：libsora和pyaudio容易安装出错，这里介绍解决办法。


libsora安装失败解决办法，使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/) ，windows的可以下载zip压缩包，方便解压。
```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现`libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如`pip install librosa==0.6.3`

安装ffmpeg， 下载地址：[http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/) ，笔者下载的是64位，static版。
然后到C盘，笔者解压，修改文件名为`ffmpeg`，存放在`C:\Program Files\`目录下，并添加环境变量`C:\Program Files\ffmpeg\bin`

最后修改源码，路径为`C:\Python3.7\Lib\site-packages\audioread\ffdec.py`，修改32行代码，如下：
```python
COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
```

pyaudio安装失败解决办法，在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)


# 创建数据
本教程笔者使用的是[中文语音语料数据集](https://github.com/KuangDD/zhvoice) ，这个数据集一共有3242个人的语音数据，有1130000+条语音数据。如果读者有其他更好的数据集，可以混合在一起使用，但要用python的工具模块aukit处理音频，降噪和去除静音。

首先是创建一个数据列表，数据列表的格式为`<语音文件路径\t语音分类标签>`，创建这个列表主要是方便之后的读取，也是方便读取使用其他的语音数据集，语音分类标签是指说话人的唯一ID，不同的语音数据集，可以通过编写对应的生成数据列表的函数，把这些数据集都写在同一个数据列表中。

在`create_data.py`写下以下代码，因为[中文语音语料数据集](https://github.com/KuangDD/zhvoice) 这个数据集是mp3格式的，作者发现这种格式读取速度很慢，所以笔者把全部的mp3格式的音频转换为wav格式。
```python
def get_data_list(infodata_path, list_path, zhvoice_path):
    with open(infodata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    sound_sum = 0
    speakers = []
    speakers_dict = {}
    for line in tqdm(lines):
        line = json.loads(line.replace('\n', ''))
        duration_ms = line['duration_ms']
        if duration_ms < 1300:
            continue
        speaker = line['speaker']
        if speaker not in speakers:
            speakers_dict[speaker] = len(speakers)
            speakers.append(speaker)
        label = speakers_dict[speaker]
        sound_path = os.path.join(zhvoice_path, line['index'])
        save_path = "%s.wav" % sound_path[:-4]
        if not os.path.exists(save_path):
            try:
                wav = AudioSegment.from_mp3(sound_path)
                wav.export(save_path, format="wav")
                os.remove(sound_path)
            except Exception as e:
                print('数据出错：%s, 信息：%s' % (sound_path, e))
                continue
        if sound_sum % 200 == 0:
            f_test.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        else:
            f_train.write('%s\t%d\n' % (save_path.replace('\\', '/'), label))
        sound_sum += 1

    f_test.close()
    f_train.close()

if __name__ == '__main__':
    get_data_list('dataset/zhvoice/text/infodata.json', 'dataset', 'dataset/zhvoice')
```

在创建数据列表之后，可能有些数据的是错误的，所以我们要检查一下，将错误的数据删除。
```python
def remove_error_audio(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = []
    for line in tqdm(lines):
        audio_path, _ = line.split('\t')
        try:
            spec_mag = load_audio(audio_path)
            lines1.append(line)
        except Exception as e:
            print(audio_path)
            print(e)
    with open(data_list_path, 'w', encoding='utf-8') as f:
        for line in lines1:
            f.write(line)


if __name__ == '__main__':
    remove_error_audio('dataset/train_list.txt')
    remove_error_audio('dataset/test_list.txt')
```

执行程序，生成数据列表。
```shell
python create_data.py
```


# 训练模型
创建`train.py`开始训练模型，搭建一个ResNet50V2分类模型，`input_shape`设置为`(257, 257, 1)`，可以通过修改第二个值修改输入音频的长度。`class_dim`为分类的总数，中文语音语料数据集一共有3242个人的语音数据，所以这里分类总数为3242，可以使用之前训练过的权重初始化模型，可以使用本项目提供的模型执行预训练。
```python
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取模型
    model = tf.keras.models.Sequential([
        tf.keras.applications.ResNet50V2(include_top=False, weights=None, input_shape=input_shape),
        tf.keras.layers.GlobalAveragePooling2D(name='feature_output'),
        tf.keras.layers.Dense(units=args.num_classes, activation=tf.nn.softmax)])

    # 打印模型
    model.summary()

    # 定义优化方法
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # 获取训练和测试数据
    train_dataset = reader.train_reader(data_list_path=args.train_list_path,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        spec_len=input_shape[1])
    test_dataset = reader.test_reader(data_list_path=args.test_list_path,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      spec_len=input_shape[1])

    # 加载预训练模型
    if args.pretrained_model is not None:
        model.load_weights(args.pretrained_model)
```

开始执行训练，使用的损失函数为交叉熵损失函数，每训练200个batch执行一次测试和保存模型，包括预测模型和网络权重。
```python
   # 开始训练
    for epoch_id in range(args.num_epoch):
        for batch_id, (sounds, labels) in enumerate(train_dataset):
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

            if batch_id % 100 == 0:
                print("Epoch %d, Batch %d, Loss %f, Accuracy %f" % (epoch_id, batch_id, train_loss.numpy(), train_accuracy))

        # 评估模型
        test_loss, test_accuracy = test(model, test_dataset)
        print('=================================================')
        print("Test %d, Loss %f, Accuracy %f" % (epoch_id, test_loss, test_accuracy))
        print('=================================================')

        # 保存模型
        save_model(model)
```

在每次训练结束之后，执行一次模型评估，使用测试集测试模型的准确率。然后并保存一次模型，分别保存了整个模型和参数，以及单独保存模型参数用于之后昨晚 预训练模型。
```python
# 评估模型
def test(model, test_dataset):
    test_losses = list()
    test_accuracies = list()
    for sounds, labels in test_dataset:
        test_result = model(sounds)
        # 获取损失值
        test_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, test_result)
        test_loss = tf.reduce_mean(test_loss)
        test_losses.append(test_loss.numpy())
        # 获取准确率
        test_accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, test_result)
        test_accuracy = np.sum(test_accuracy.numpy()) / len(test_accuracy.numpy())
        test_accuracies.append(test_accuracy)
    return sum(test_losses) / len(test_losses), sum(test_accuracies) / len(test_accuracies)


# 保存模型
def save_model(model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    model.save(filepath=os.path.join(args.save_model, 'infer_model.h5'))
    model.save_weights(filepath=os.path.join(args.save_model, 'model_weights.h5'))
```

# 声纹对比
下面开始实现声纹对比，创建`infer_contrast.py`程序，在加载模型时，不要直接加载整个模型，而是加载模型的最后分类层的上一层，也就是我们定义名称为`feature_output`的池化层，这样就可以获取到语音的特征数据。这里顺便介绍个工具，通过使用[netron](https://github.com/lutzroeder/netron)查看每一层的输入和输出的名称。

```python
# 加载模型，根据名称获取分类输出的上一层特征输出
layer_name = 'feature_output'
model = tf.keras.models.load_model(args.model_path)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 获取均值和标准值
input_shape = eval(args.input_shape)


# 预测音频
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
    feature = intermediate_layer_model.predict(data)
    return feature[0]

```

有了上面的`infer()`预测函数之后，在这个加载数据函数中并没有限定输入音频的大小，但是裁剪静音后的音频不能小于1.3秒，这样就可以输入任意长度的音频。执行预测之后数据的是语音的特征值。使用预测函数，我们输入两个语音，通过预测函数获取他们的特征数据，使用这个特征数据可以求他们的对角余弦值，得到的结果可以作为他们相识度。对于这个相识度的阈值，读者可以根据自己项目的准确度要求进行修改。
```python
if __name__ == '__main__':
    # 要预测的两个人的音频文件
    feature1 = infer(args.audio_path1)
    feature2 = infer(args.audio_path2)
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print("%s 和 %s 为同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (args.audio_path1, args.audio_path2, dist))
```

# 声纹识别
在上面的声纹对比的基础上，我们创建`infer_recognition.py`实现声纹识别。同样是使用上面声纹对比的预测函数`infer()`，通过这两个同样获取语音的特征数据。
```python
# 加载模型，根据名称获取分类输出的上一层特征输出
layer_name = 'feature_output'
model = tf.keras.models.load_model(args.model_path)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 获取均值和标准值
input_shape = eval(args.input_shape)

person_feature = []
person_name = []


# 预测音频
def infer(audio_path):
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[2])
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
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)[0]
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
            if p > args.threshold:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        except:
            pass
```
