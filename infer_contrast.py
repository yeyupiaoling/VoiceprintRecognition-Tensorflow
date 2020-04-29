import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


layer_name = 'dense'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_len = 32640
    # 裁剪过长的音频，过短的补0
    if len(wav_output) > wav_len:
        wav_output = wav_output[:wav_len]
    else:
        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
    wav_output = np.array(wav_output)
    # 获取梅尔频谱
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, np.newaxis, ...]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature


if __name__ == '__main__':
    # 要预测的两个人的音频文件
    person1 = 'dataset/UrbanSound8K/audio/fold8/193699-2-0-46.wav'
    person2 = 'dataset/UrbanSound8K/audio/fold8/193699-2-0-46.wav'
    feature1 = infer(person1)
    feature2 = infer(person2)
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.9:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
