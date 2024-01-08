import random

import tensorflow as tf
import librosa
import numpy as np


# 加载并预处理音频
def load_audio(audio_path, mode='train', sr=16000, spec_len=257, use_audio_len=2.6):
    # 读取音频数据
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=wav, sr=sr)
    assert duration >= use_audio_len / 2., f"非静音部分长度不能低于{use_audio_len}s"
    # 裁剪音频，在这里裁剪的原因是避免音频太长，导致计算音频特征耗时过长
    crop_len = int(use_audio_len * sr_ret) if duration >= use_audio_len else int(duration * sr_ret)
    audio_len = int(duration * sr_ret)
    if mode == 'train':
        start = random.randint(0, int(audio_len - crop_len))
        wav = wav[start:start + crop_len]
    else:
        wav = wav[:crop_len]
    # 数据拼接，对于音频长度不够的，反转音频拼接
    if mode == 'train':
        if duration < use_audio_len:
            extended_wav = np.append(wav, wav)
        else:
            extended_wav = wav
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        if duration < use_audio_len:
            extended_wav = np.append(wav, wav[::-1])
        else:
            extended_wav = wav
    # 计算短时傅里叶变换
    linear = librosa.stft(extended_wav, n_fft=512, win_length=400, hop_length=160)
    mag, _ = librosa.magphase(linear)
    freq, freq_time = mag.shape
    assert freq_time >= spec_len, f"特征长度必须大于等于{spec_len}，当前为：{freq_time}"
    spec_mag = mag[:, :spec_len]
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
    ds = tf.data.Dataset.from_generator(generator=data_generator,
                                        args=(data_list_path, spec_len),
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
