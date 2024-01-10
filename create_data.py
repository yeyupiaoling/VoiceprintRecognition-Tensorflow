import json
import os
import random
import sys

from tqdm import tqdm

from utils.reader import load_audio


# 生成数据列表
def get_data_list(infodata_path, list_path, zhvoice_path):
    with open(infodata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_enroll = open(os.path.join(list_path, 'enroll_list.txt'), 'w')
    f_trials = open(os.path.join(list_path, 'trials_list.txt'), 'w')

    speakers_name = set()
    for line in lines:
        line = json.loads(line.replace('\n', ''))
        duration_ms = line['duration_ms']
        if duration_ms < 1300:
            continue
        sound_path = os.path.join(zhvoice_path, line['index'])
        if not os.path.exists(sound_path):
            continue
        speaker = line['speaker']
        speakers_name.add(speaker)
    speakers_name = list(speakers_name)
    test_speaker_name = [name for i, name in enumerate(speakers_name) if i % 32 == 0]
    train_speaker_name = [name for name in speakers_name if name not in test_speaker_name]
    train_speaker_dict = {name: i for i, name in enumerate(train_speaker_name)}
    test_speaker_dict = {name: i for i, name in enumerate(test_speaker_name)}
    print(f'训练集有{len(train_speaker_name)}个说话人，测试集有{len(test_speaker_name)}个说话人')

    test_data = {i: [] for i in range(len(test_speaker_name))}
    for line in tqdm(lines):
        line = json.loads(line.replace('\n', ''))
        duration_ms = line['duration_ms']
        if duration_ms < 1300:
            continue
        speaker = line['speaker']
        sound_path = os.path.join(zhvoice_path, line['index'])
        if not os.path.exists(sound_path):
            continue
        if speaker in test_speaker_name:
            speaker_id = test_speaker_dict[speaker]
            test_data[speaker_id].append(sound_path.replace('\\', '/'))
        if speaker in train_speaker_name:
            speaker_id = train_speaker_dict[speaker]
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), speaker_id))
    f_train.close()

    for data in test_data.items():
        speaker_id, data = data
        for i, d in enumerate(data):
            if i == 0:
                f_enroll.write('%s\t%d\n' % (d, speaker_id))
            else:
                f_trials.write('%s\t%d\n' % (d, speaker_id))


# 删除错误音频
def remove_error_audio(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines1 = []
    for line in tqdm(lines):
        audio_path, _ = line.split('\t')
        try:
            _ = load_audio(audio_path)
            lines1.append(line)
        except Exception as e:
            print(audio_path, file=sys.stderr)
            print(e, file=sys.stderr)
    with open(data_list_path, 'w', encoding='utf-8') as f:
        for line in lines1:
            f.write(line)


if __name__ == '__main__':
    get_data_list('dataset/zhvoice/text/infodata.json', 'dataset', 'dataset/zhvoice')
    remove_error_audio('dataset/train_list.txt')
    remove_error_audio('dataset/enroll_list.txt')
    remove_error_audio('dataset/trials_list.txt')
