import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


layer_name = 'dropout_1'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


# 读取音频数据
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


if __name__ == '__main__':
    # 要预测的两个人的音频文件
    person1 = 'audio_db/wanwen1.wav'
    person2 = 'audio_db/wanwen2.wav'
    feature1 = infer(person1)[0]
    feature2 = infer(person2)[0]
    # 对角余弦值
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > 0.7:
        print("%s 和 %s 为同一个人，相似度为：%f" % (person1, person2, dist))
    else:
        print("%s 和 %s 不是同一个人，相似度为：%f" % (person1, person2, dist))
