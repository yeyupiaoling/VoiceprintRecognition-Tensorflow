import argparse
import functools

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments, compute_eer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('enroll_list',      str,    'dataset/enroll_list.txt',    '测试数据的数据列表路径')
add_arg('trials_list',      str,    'dataset/trials_list.txt',    '测试数据的数据列表路径')
add_arg('input_shape',      str,    '(1, 257, 257)',              '数据输入的形状')
add_arg('model_path',       str,    'models/best_model/infer_model.h5',  '预测模型的路径')
args = parser.parse_args()

print_arguments(args)

# 加载模型
model = tf.keras.models.load_model(args.model_path)
model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization').output)


# 预测音频
def infer(audio_path):
    input_shape = eval(args.input_shape)
    data = load_audio(audio_path, mode='test', spec_len=input_shape[2])
    data = data[np.newaxis, :]
    # 执行预测
    feature = model.predict(data)
    return feature[0]


def get_all_audio_feature(list_path):
    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    features, labels = [], []
    print('开始提取全部的音频特征...')
    for line in tqdm(lines):
        path, label = line.replace('\n', '').split('\t')
        feature = infer(path)
        features.append(feature)
        labels.append(int(label))
    return features, labels


def main():
    enroll_features, enroll_labels = get_all_audio_feature(args.list_path)
    trials_features, trials_labels = get_all_audio_feature(args.list_path)
    print('开始对比音频特征...')
    all_score, all_labels = [], []
    for i in tqdm(range(len(trials_features)), desc='特征对比'):
        trials_feature = np.expand_dims(trials_features[i], 0).repeat(len(enroll_features), axis=0)
        score = cosine_similarity(trials_feature, enroll_features).tolist()[0]
        trials_label = np.expand_dims(trials_labels[i], 0).repeat(len(enroll_features), axis=0)
        y_true = np.array(enroll_labels == trials_label).astype(np.int32).tolist()
        all_score.extend(score)
        all_labels.extend(y_true)
    y_score = np.asarray(all_score)
    y_true = np.asarray(all_labels)
    eer, eer_threshold = compute_eer(y_true, y_score)
    print(f'【EER】 threshold: {eer_threshold:.5f}，EER: {eer:.5f}')


if __name__ == '__main__':
    main()
