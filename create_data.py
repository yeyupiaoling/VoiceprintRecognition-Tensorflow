import json
import os
from random import sample
from tqdm import tqdm
import librosa
import numpy as np
from pydub import AudioSegment


# 生成数据列表
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


# 计算均值和标准值
def compute_mean_std(data_list_path='dataset/train_list.txt', output_path='dataset/mean_std.npy', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = sample(lines, 5000)
    data = None
    for line in tqdm(lines):
        audio_path, _ = line.split('\t')
        wav, sr_ret = librosa.load(audio_path, sr=sr)
        extended_wav = np.append(wav, wav[::-1])
        linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        linear_T = linear.T
        mag, _ = librosa.magphase(linear_T)
        mag_T = mag.T
        spec_mag = mag_T[:, :spec_len]
        if data is None:
            data = np.array(spec_mag, dtype='float32')
        else:
            data = np.vstack((data, spec_mag))
    mean = np.mean(data, 0, keepdims=True)
    std = np.std(data, 0, keepdims=True)
    np.save(output_path, [mean, std])


# 删除错误音频
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
    get_data_list('dataset/zhvoice/text/infodata.json', 'dataset', 'dataset/zhvoice')
    compute_mean_std('dataset/train_list.txt')
    remove_error_audio('dataset/train_list.txt')
    remove_error_audio('dataset/test_list.txt')
