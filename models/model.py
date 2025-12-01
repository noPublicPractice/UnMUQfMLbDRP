import json
import os
import pickle
import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils.utils import (
    logger,
    calc_accuracy,
    calc_f1,
    calc_ppv,
    calc_tpr,
    calc_pred_pro_weighted_sum,
    group_values_by_density,
    load_json_data,
    save_json_data,
    test_data_label_index,
)


def _select_model(model_name: str):
    if model_name == 'GBDT':
        return GradientBoostingClassifier(), 'GBDT'
    if model_name == 'XGBt':
        return XGBClassifier(), 'XGBt'
    return LGBMClassifier(), 'LGBM'


def _log_duration(label: str, start_time: float):
    duration = time.time() - start_time
    logger.info("%s elapsed time: %f seconds", label, duration)
    return time.time()


def _get_vectorizer(cfg):
    return TfidfVectorizer(analyzer="word", max_features=cfg.max_features_num, dtype=np.float32)


def _get_label_encoder():
    return LabelEncoder()


def _get_model_path(cfg, fold_index: int, model_name: str) -> str:
    return cfg.model_values_pkl_name % (cfg.label, cfg.folds, cfg.great_random_state, fold_index, model_name)


def _train_single_fold(cfg, fold: int, vectorizer, model, evaluation_path: str):
    pickle_path = cfg.data_frame_pickle_pat % (cfg.label, cfg.folds, cfg.great_random_state, fold)
    train_data_frame, test_data_frame = pickle.load(open(pickle_path, "rb"))
    train_obs = vectorizer.fit_transform(train_data_frame['OBSERVATION'])
    test_obs = vectorizer.transform(test_data_frame['OBSERVATION'])
    label_encoder = _get_label_encoder()
    train_labels = label_encoder.fit_transform(train_data_frame[cfg.label_ch])
    test_labels = label_encoder.transform(test_data_frame[cfg.label_ch])

    start_time = time.time()
    model.fit(train_obs, train_labels)
    _log_duration('model', start_time)
    pred_pro = model.predict_proba(test_obs)[:, 1]
    pred_lr = model.predict(test_obs)

    joblib.dump((vectorizer, model, test_labels, pred_pro, pred_lr), open(evaluation_path, "wb"))
    _write_model_evaluation(cfg, fold, vectorizer, model, test_labels, pred_pro, pred_lr)


def _write_model_evaluation(cfg, exclude_fold: int, vectorizer, model, test_labels, pred_pro, pred_lr):
    false_positive_rate, true_positive_rate, _ = roc_curve(test_labels, pred_pro)
    au_roc = round(auc(false_positive_rate, true_positive_rate), 6)
    sensitivity = round(recall_score(test_labels, pred_lr), 6)
    specificity = round(1 - recall_score(1 - test_labels, pred_lr), 6)
    important_feature = _get_important_feature(vectorizer, model, 4, cfg.label)
    with open(cfg.evaluation_output, 'a', encoding='utf-8') as fp:
        fp.write(
            '%d\t\t\t%d\t\t%s\t%f\t\t%f\t\t%f\t%s\n'
            % (cfg.great_random_state, exclude_fold, cfg.label, sensitivity, specificity, au_roc, important_feature)
        )


def _get_important_feature(vectorizer, model, values_num: int, label: str) -> str:
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = model.feature_importances_
    explaining_df = pd.DataFrame({"feature_names": feature_names, "score": feature_importance})
    explaining_df_sort = explaining_df.sort_values(by="score", ascending=False)
    explaining_values = explaining_df_sort.values
    for i in range(min(values_num, explaining_values.shape[0])):
        explaining_values[i][1] = round(explaining_values[i][1], 5)
    return str(explaining_values[:values_num]).replace('\n', '').replace('\'', '')


def train_ensemble_models(cfg):
    os.makedirs(os.path.dirname(cfg.evaluation_output), exist_ok=True) if os.path.dirname(cfg.evaluation_output) else None
    vectorizer_template = _get_vectorizer(cfg)
    for fold in range(1, cfg.folds + 1):
        model, model_name = _select_model(cfg.model_name)
        model_path = _get_model_path(cfg, fold, model_name)
        if os.path.exists(model_path):
            logger.info("Model already exists, skip Fold %d: %s", fold, model_path)
            continue
        vectorizer = pickle.loads(pickle.dumps(vectorizer_template))
        _train_single_fold(cfg, fold, vectorizer, model, model_path)


def reevaluate_models(cfg):
    for fold in range(1, cfg.folds + 1):
        model_path = _get_model_path(cfg, fold, cfg.model_name)
        if os.path.exists(model_path):
            vectorizer, model, test_labels, pred_pro, pred_lr = joblib.load(open(model_path, "rb"))
            _write_model_evaluation(cfg, fold, vectorizer, model, test_labels, pred_pro, pred_lr)
        else:
            logger.warning("Model file does not exist: %s", model_path)


def _load_predictions(cfg, label_en: str, random_state: int) -> Tuple[np.ndarray, List[List[float]]]:
    pred_pro_list = []
    test_data_label = None
    for fold in range(cfg.folds):
        joblib_name = cfg.model_values_joblib_name % (label_en, cfg.folds, random_state, fold + 1, cfg.model_name)
        vectorizer, model, test_labels, pred_pro, pred_lr = joblib.load(open(joblib_name, "rb"))
        pred_pro_list.append(list(pred_pro))
        test_data_label = test_labels
    return test_data_label, pred_pro_list


def generate_threshold_metrics(cfg):
    for label_ch, label_en, random_state in zip(cfg.label_ch_list, cfg.label_en_list, cfg.great_random_state):
        test_data_label, pred_pro_list = _load_predictions(cfg, label_en, random_state)
        positive_index, negative_index = test_data_label_index(test_data_label)
        pred_pro_weighted_sum = calc_pred_pro_weighted_sum(cfg.au_roc, label_en, pred_pro_list)
        positive_group_dict = group_values_by_density(pred_pro_weighted_sum[positive_index], cfg.density)
        negative_group_dict = group_values_by_density(pred_pro_weighted_sum[negative_index], cfg.density)
        threshold_with_assessed_values = _calc_threshold_with_assessed_values(
            positive_group_dict, negative_group_dict, len(positive_index), len(negative_index), cfg
        )
        json_path = cfg.do_write_json_path % (label_en, label_ch, cfg.density)
        save_json_data(json_path, threshold_with_assessed_values)
        logger.info("Threshold evaluation data saved: %s", json_path)


def _calc_threshold_with_assessed_values(positive_group, negative_group, positive_n, negative_n, cfg):
    tp, fn = positive_n, 0
    fp, tn = negative_n, 0
    amount_sum = tp + fn + fp + tn
    threshold_interval = 1 / cfg.density
    classify_threshold = np.zeros(cfg.density + 1)
    acc = np.zeros(cfg.density + 1)
    ppv = np.zeros(cfg.density + 1)
    tpr = np.zeros(cfg.density + 1)
    f1 = np.zeros(cfg.density + 1)

    for index, threshold in enumerate(np.linspace(0.0, 1.0, cfg.density + 1)):
        if index:
            group_name = '%.3f-%.3f' % (threshold - threshold_interval, threshold)
            if group_name in positive_group:
                tp -= positive_group[group_name]
                fn += positive_group[group_name]
            if group_name in negative_group:
                fp -= negative_group[group_name]
                tn += negative_group[group_name]
        classify_threshold[index] = round(threshold, 3)
        acc[index] = calc_accuracy(tp, tn, amount_sum)
        ppv[index] = calc_ppv(tp, fp)
        tpr[index] = calc_tpr(tp, fn)
        f1[index] = calc_f1(tp, fn, fp)
    return {
        "classify_threshold": list(classify_threshold),
        "acc": list(acc),
        "ppv": list(ppv),
        "tpr": list(tpr),
        "acc_multiply_tpr": list(acc * tpr),
        "f1": list(f1),
    }


def print_threshold_evaluation(cfg):
    for label_ch, label_en, random_state in zip(cfg.label_ch_list, cfg.label_en_list, cfg.great_random_state):
        json_path = cfg.threshold_with_assessed_values_path % (label_en, label_ch, cfg.density)
        assessed_values = load_json_data(json_path)
        max_f1_index = assessed_values['f1'].index(max(assessed_values['f1']))
        great_threshold = assessed_values['classify_threshold'][max_f1_index]
        _, negative_index, positive_index = _load_eval_groups(cfg, label_en, random_state)
        logger.info("%s best threshold %.3f (index %d)", label_ch, great_threshold, max_f1_index)
        _print_threshold_scores(assessed_values, max_f1_index)


def _load_eval_groups(cfg, label_en, random_state):
    joblib_name = cfg.model_values_joblib_name % (label_en, cfg.folds, random_state, 1, cfg.model_name)
    vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(joblib_name, "rb"))
    positive_index, negative_index = test_data_label_index(test_data_label)
    return test_data_label, negative_index, positive_index


def _print_threshold_scores(assessed_values, index):
    metrics = ['acc', 'ppv', 'tpr', 'f1']
    for metric in metrics:
        logger.info("%s: %.6f", metric.upper(), assessed_values[metric][index])


def compute_statistical_coefficients(cfg):
    for label_ch, label_en, random_state in zip(cfg.label_ch_list, cfg.label_en_list, cfg.great_random_state):
        test_data_label, pred_pro_list = _load_predictions(cfg, label_en, random_state)
        positive_index, negative_index = test_data_label_index(test_data_label)
        pred_pro_weighted_sum = calc_pred_pro_weighted_sum(cfg.au_roc, label_en, pred_pro_list)
        positive_group_dict = group_values_by_density(pred_pro_weighted_sum[positive_index], cfg.interval_num)
        negative_group_dict = group_values_by_density(pred_pro_weighted_sum[negative_index], cfg.interval_num)
        statistical_coefficients = _update_statistical_coefficients(cfg, positive_group_dict, negative_group_dict)
        json_path = cfg.do_write_json_path % (label_en, label_ch, cfg.interval_num)
        save_json_data(json_path, statistical_coefficients)
        logger.info("Statistical coefficient data saved: %s", json_path)


def _update_statistical_coefficients(cfg, positive_group_dict, negative_group_dict):
    stats = {"partial_positive_pro": {}, "partial_negative_pro": {}}
    threshold_spacing = 1 / cfg.interval_num
    for index, interval_right in enumerate(np.linspace(threshold_spacing, 1.0, cfg.interval_num)):
        interval_left = interval_right - threshold_spacing
        group_name = '%.3f-%.3f' % (interval_left, interval_right)
        pos_num = positive_group_dict.get(group_name, 0)
        neg_num = negative_group_dict.get(group_name, 0)
        total = pos_num + neg_num
        if total == 0:
            stats["partial_positive_pro"][group_name] = 1.0
            stats["partial_negative_pro"][group_name] = 1.0
        else:
            stats["partial_positive_pro"][group_name] = pos_num / total
            stats["partial_negative_pro"][group_name] = neg_num / total
    return stats


def prepare_confidence_artifacts(cfg):
    for label_en, label_ch, random_state in zip(cfg.label_en_list, cfg.label_ch_list, cfg.great_random_state):
        test_data_label, pred_pro_list = _load_predictions(cfg, label_en, random_state)
        pred_pro_weighted_sum, pred_pro_confidence_score, true_index, false_index, test_data_label_arr, pred_label_arr, great_threshold = \
            _compute_confidence(cfg, pred_pro_list, test_data_label, label_en, label_ch)
        _write_confidence_json(cfg, label_en, label_ch, pred_pro_weighted_sum, pred_pro_confidence_score, true_index, false_index)
        _write_predict_information_json(
            cfg,
            label_en,
            label_ch,
            pred_pro_weighted_sum,
            pred_pro_confidence_score,
            true_index,
            false_index,
            test_data_label_arr,
            pred_label_arr,
            great_threshold,
        )


def _compute_confidence(cfg, pred_pro_list, test_data_label, label_en, label_ch):
    _, great_threshold = _get_classify_threshold(cfg, label_en, label_ch)
    partial_proportion_dict = _get_partial_proportion(cfg, label_en, label_ch, great_threshold)
    pred_pro_arr, confidence_factor, pred_pro_weighted_sum_sorted, test_data_label_arr = \
        _group_partial_proportion(cfg, label_en, pred_pro_list, test_data_label, partial_proportion_dict)
    pred_confidence = _get_confidence_score(cfg, pred_pro_arr, confidence_factor)
    true_index, false_index, pred_label_arr = _get_true_false_index(pred_pro_weighted_sum_sorted, great_threshold, test_data_label_arr)
    return pred_pro_weighted_sum_sorted, pred_confidence, true_index, false_index, test_data_label_arr, pred_label_arr, great_threshold


def _get_classify_threshold(cfg, label_en, label_ch):
    json_path = cfg.threshold_with_assessed_values_path % (label_en, label_ch, cfg.density)
    assessed_values = load_json_data(json_path)
    max_f1_index = assessed_values['f1'].index(max(assessed_values['f1']))
    great_threshold = assessed_values['classify_threshold'][max_f1_index]
    return max_f1_index, great_threshold


def _get_partial_proportion(cfg, label_en, label_ch, great_threshold):
    json_path = cfg.statistical_coefficients_path % (label_en, label_ch, cfg.interval_num)
    partial_proportion = load_json_data(json_path)
    threshold_interval = 1 / cfg.density
    partial_proportion_with_threshold = {}
    flag = True
    for interval_right in np.linspace(threshold_interval, 1.0, cfg.density):
        interval_left = interval_right - threshold_interval
        group_name = '%.3f-%.3f' % (interval_left, interval_right)
        if flag:
            now_value = partial_proportion['partial_negative_pro'].get(group_name, 0.0)
        else:
            if cfg.confidence_score_f == 'MCF×ISP':
                now_value = partial_proportion['partial_positive_pro'].get(group_name, 0.0)
            else:
                temp_k = cfg.temp_coefficient_k[label_en] * interval_right + cfg.temp_coefficient_b[label_en]
                temp_b = cfg.temp_const[label_en] * (1 - interval_right)
                now_value = temp_k * partial_proportion['partial_positive_pro'].get(group_name, 0.0) + temp_b
        partial_proportion_with_threshold[group_name] = now_value
        if flag and group_name.endswith('%.3f' % great_threshold):
            flag = False
    return partial_proportion_with_threshold


def _group_partial_proportion(cfg, label_en, pred_pro_list, test_data_label, partial_proportion_dict):
    confidence_factor_with_partial = []
    pred_weighted = calc_pred_pro_weighted_sum(cfg.au_roc, label_en, pred_pro_list)
    pred_pro_list = pred_pro_list.copy()
    pred_pro_list.append(list(pred_weighted))
    pred_pro_list.append(list(test_data_label))
    pred_pro_arr_extend = np.array(pred_pro_list)
    pred_pro_arr_extend_t = pred_pro_arr_extend.T
    pred_pro_sorted = pred_pro_arr_extend_t[np.argsort(pred_pro_arr_extend_t[:, cfg.folds])]
    pred_pro_arr = pred_pro_sorted.T[:cfg.folds]
    pred_weighted_sorted = pred_pro_sorted.T[cfg.folds]
    test_data_label_arr = pred_pro_sorted.T[cfg.folds + 1].astype(int)
    for group_key, pred_group in group_values_by_density(pred_weighted_sorted, cfg.density).items():
        pred_count = pred_group
        partial_value = partial_proportion_dict[group_key]
        confidence_factor_with_partial.extend([partial_value] * pred_count)
    return pred_pro_arr, confidence_factor_with_partial, pred_weighted_sorted, test_data_label_arr


def _get_confidence_score(cfg, pred_pro_arr, confidence_factor_with_partial):
    pred_pro_arr = pred_pro_arr.T
    confidence_factor_with_mcf = 1 - np.std(pred_pro_arr, axis=1)
    if cfg.confidence_score_f in ['MCF×ISP', 'MCF×SCF']:
        return confidence_factor_with_mcf * np.array(confidence_factor_with_partial)
    return confidence_factor_with_mcf


def _get_true_false_index(pred_weighted_sum, great_threshold, test_data_label_arr):
    binarizer = (pred_weighted_sum >= great_threshold).astype(int)
    true_index = np.where(test_data_label_arr == binarizer)[0]
    false_index = np.where(test_data_label_arr != binarizer)[0]
    return true_index, false_index, binarizer


def _write_confidence_json(cfg, label_en, label_ch, pred_weighted_sum, pred_confidence, true_index, false_index):
    saved_dict = {
        "true": {
            "pred_pro_weighted_sum": list(pred_weighted_sum[true_index]),
            "pred_pro_confidence": list(pred_confidence[true_index]),
        },
        "false": {
            "pred_pro_weighted_sum": list(pred_weighted_sum[false_index]),
            "pred_pro_confidence": list(pred_confidence[false_index]),
        },
    }
    json_path = cfg.do_write_confidence_score_json_path % (label_en, label_ch, cfg.confidence_score_f)
    save_json_data(json_path, saved_dict)
    logger.info("Confidence score data saved: %s", json_path)


def _write_predict_information_json(cfg, label_en, label_ch, pred_weighted_sum, pred_confidence, true_index,
                                    false_index, test_data_label_arr, pred_label_arr, threshold):
    saved_dict = {
        "pred_pro_weighted_sum": list(pred_weighted_sum),
        "pred_pro_confidence_score": list(pred_confidence),
        "true_predict_index": list(true_index.astype(float)),
        "false_predict_index": list(false_index.astype(float)),
        "test_data_label_arr": list(test_data_label_arr.astype(float)),
        "pred_label_arr": list(pred_label_arr.astype(float)),
        "great_classify_threshold": threshold,
    }
    json_path = cfg.do_write_predict_information_json_path % (label_en, label_ch, cfg.confidence_score_f)
    save_json_data(json_path, saved_dict)
    logger.info("Prediction information data saved: %s", json_path)

