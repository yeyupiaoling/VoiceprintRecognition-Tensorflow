import random

import tensorflow as tf
import librosa
import numpy as np
from aukit import remove_silence, remove_noise


# 加载并预处理音频
def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    # 推理的数据要移除静音部分
    if mode == 'infer':
        wav = remove_silence(wav, sr)
        wav = remove_noise(wav, sr)
    # 数据拼接
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    # 计算短时傅里叶变换
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    linear_T = linear.T
    mag, _ = librosa.magphase(linear_T)
    mag_T = mag.T
    freq, freq_time = mag_T.shape
    assert freq_time >= spec_len, "非静音部分长度不能低于1.3s"
    if mode == 'train':
        # 随机裁剪
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag_T[:, rand_time:rand_time + spec_len]
    else:
        spec_mag = mag_T[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[:, :, np.newaxis]
    return spec_mag


# 预处理数据
def data_generator(data_list_path, spec_len=257):
    with open(data_list_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    for line in lines:
        audio_path, label = line.replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode='train', spec_len=spec_len)
        yield spec_mag, np.array(int(label))


# 读取训练数据
def train_reader(data_list_path, batch_size, num_epoch, spec_len=257):
    ds = tf.data.Dataset.from_generator(generator=lambda:data_generator(data_list_path, spec_len=spec_len),
                                        output_types=(tf.float32, tf.int64))

    train_dataset = ds.shuffle(buffer_size=1000) \
        .batch(batch_size=batch_size) \
        .repeat(num_epoch) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset


# 读取测试数据
def test_reader(data_list_path, batch_size, spec_len=257):
    ds = tf.data.Dataset.from_generator(generator=lambda:data_generator(data_list_path, spec_len=spec_len),
                                        output_types=(tf.float32, tf.int64))

    test_dataset = ds.batch(batch_size=batch_size)
    return test_dataset
