import json
import logging
import os
from collections import OrderedDict
from typing import Dict, Iterable, List

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("project")


def calc_accuracy(tp, tn, amount_sum):
    return (tp + tn) / amount_sum if amount_sum else 0.0


def calc_ppv(tp, fp):
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def calc_tpr(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def calc_f1(tp, fn, fp):
    denominator = 2 * tp + fn + fp
    if denominator == 0:
        return 0.0
    return 2 * tp / denominator


def test_data_label_index(test_data_label):
    label_list = np.array(test_data_label)
    zeros_list = np.zeros(len(test_data_label))
    positive_index = np.where(label_list != zeros_list)[0]
    negative_index = np.where(label_list == zeros_list)[0]
    return positive_index, negative_index


def calc_pred_pro_weighted_sum(au_roc: Dict[str, List[float]], label_en: str, pred_pro_list: List[List[float]]):
    pred_pro_k = np.array(au_roc[label_en]) / sum(au_roc[label_en])
    pred_pro_arr = np.array(pred_pro_list)
    return np.dot(pred_pro_k, pred_pro_arr)


def group_values_by_density(values: Iterable[float], density: int) -> OrderedDict:
    grouped = OrderedDict()
    sorted_values = sorted(values)
    for value in sorted_values:
        key = int(value * density)
        group_name = '%.3f-%.3f' % (key / density, (key + 1) / density)
        grouped[group_name] = grouped.get(group_name, 0) + 1
    return grouped


def save_json_data(path: str, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def load_json_data(path: str):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)

