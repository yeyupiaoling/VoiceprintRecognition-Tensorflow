import os
import wave
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
from tensorflow.keras.models import Model


layer_name = 'dense'
model = tf.keras.models.load_model('models/resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

person_feature = []
person_name = []


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    # 裁剪过长的音频，过短的补0
    if len(wav_output) > 65489:
        wav_output = wav_output[:65489]
    else:
        wav_output.extend(np.zeros(shape=[65489 - len(wav_output)], dtype=np.float32))
    wav_output = np.array(wav_output)
    # 获取梅尔频谱
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr).astype(np.float32)
    ps = ps[np.newaxis, np.newaxis, ...]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)
        person_name.append(name)
        person_feature.append(feature)


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


if __name__ == '__main__':
    load_audio_db('audio_db')
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 4
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
            i = input("按下回车键开机录音，录音4秒中：")
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
            if p > 0.9:
                print("识别说话的为：%s，相似度为：%f" % (name, p))
            else:
                print("音频库没有该用户的语音")
        except:
            pass
