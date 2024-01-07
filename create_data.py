import json
import os
import sys

from tqdm import tqdm

from utils.reader import load_audio


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
        if not os.path.exists(sound_path):
            continue
        if sound_sum % 200 == 0:
            f_test.write('%s\t%d\n' % (sound_path.replace('\\', '/'), label))
        else:
            f_train.write('%s\t%d\n' % (sound_path.replace('\\', '/'), label))
        sound_sum += 1

    f_test.close()
    f_train.close()


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
    remove_error_audio('dataset/test_list.txt')
