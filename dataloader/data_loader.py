import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils.utils import logger


def prepare_patient_info(cfg) -> pd.DataFrame:
    """Filter atrial fibrillation patients from patient information and cache the result."""
    file_name = cfg.patient_pkl_p
    if not os.path.exists(file_name):
        df_icd_list = pd.read_csv(cfg.icd_text_path)
        df_icd_10 = df_icd_list[df_icd_list['icd_code'].str.contains('I48')]
        df_icd_09 = df_icd_list[df_icd_list['icd_code'].str.startswith('42731')]
        df_icd_09 = df_icd_09[df_icd_09['icd_version'] == 9]
        df_icd = pd.concat([df_icd_10, df_icd_09], axis=0, ignore_index=True)

        df_diagnoses = pd.read_csv(cfg.icd_patient_p)
        patient_info = df_diagnoses[df_diagnoses['icd_code'].isin(df_icd['icd_code'])]
        patient_info = patient_info[['subject_id', 'hadm_id']]
        # Column names are normalized to English keys for downstream processing.
        patient_info.columns = ['PATIENTID', 'STAYID']
        pickle.dump(patient_info, open(file_name, "wb"))
    else:
        logger.info("Loaded cached patient information %s", file_name)
    patient_info = pickle.load(open(file_name, "rb"))
    return patient_info


def remove_leaking_feature(value: str) -> str:
    if pd.isna(value):
        return value
    elements = value.split(',')
    filtered_elements = [el for el in elements if not any(code in el for code in ['i48', 'icd_9_42731'])]
    return ','.join(filtered_elements)


def prepare_data(cfg) -> pd.DataFrame:
    """Load raw training data and clean leakage features."""
    file_name = cfg.data_pkl_path
    if not os.path.exists(file_name):
        df_data = pd.read_csv(cfg.df_training_AF4_path, sep=';')
        df_data.drop('PREDICTION_TARGET', axis=1, errors='ignore', inplace=True)
        df_data['DIAGNOSIS_HISTORY'] = df_data['DIAGNOSIS_HISTORY'].apply(remove_leaking_feature)
        pickle.dump(df_data, open(file_name, "wb"))
    else:
        logger.info("Loaded cached training data %s", file_name)
    df_data = pickle.load(open(file_name, "rb"))
    df_data.drop('PREDICTION_TARGET', axis=1, errors='ignore', inplace=True)
    return df_data


def _split_train_test(df_data: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patient_id_unique = df_data['PATIENTID'].unique()
    patient_id_after_shuffle = shuffle(patient_id_unique, random_state=cfg.random_state_num)
    split_point = int(0.8 * len(patient_id_after_shuffle))
    patient_id_training = patient_id_after_shuffle[:split_point]
    patient_id_test_eval = patient_id_after_shuffle[split_point:]
    df_training = df_data[df_data['PATIENTID'].isin(patient_id_training)]
    df_test = df_data[df_data['PATIENTID'].isin(patient_id_test_eval)]
    return df_training.copy(), df_test.copy()


def create_train_test(cfg):
    """Create training and test sets and control the ratio of positive and negative samples."""
    df_data = prepare_data(cfg)
    patient_info = prepare_patient_info(cfg)

    # Column names are standardized to English keys for joins.
    patient_with_symptoms_predicted = df_data[df_data['STAYID'].isin(patient_info['STAYID'])].copy()
    patient_with_symptoms_predicted['PREDICTION_TARGET'] = 1
    pat_not_with_symptoms_predicted = df_data[~df_data['PATIENTID'].isin(patient_info['PATIENTID'])].copy()
    pat_not_with_symptoms_predicted['PREDICTION_TARGET'] = 0
    df_labelled = pd.concat([patient_with_symptoms_predicted, pat_not_with_symptoms_predicted])

    df_training, df_test = _split_train_test(df_labelled, cfg)
    df_test.to_csv('df_eval_AF4.csv', sep=";", index=False)

    df_training_case = df_training[df_training['PREDICTION_TARGET'] == 1]
    df_training_control = df_training[df_training['PREDICTION_TARGET'] == 0]
    df_training_control = shuffle(df_training_control, random_state=cfg.random_state_num)
    df_training_control = df_training_control[:5 * len(df_training_case)]
    df_training_balanced = pd.concat([df_training_case, df_training_control])
    df_training_balanced = shuffle(df_training_balanced, random_state=cfg.random_state_num)
    df_training_balanced.to_csv('df_training_AF4.csv', sep=";", index=False)
    logger.info("Training/test datasets updated: df_training_AF4.csv, df_eval_AF4.csv")


def generate_cross_training_csv(cfg):
    """Generate stratified cross-validation training/validation sets by patient ID."""
    whole_training_data = pd.read_csv(cfg.whole_training_csv_path % cfg.label, sep=';')
    # Filter out low-quality admission year groups.
    whole_training_data = whole_training_data[~(whole_training_data['ADMISSION_YEAR_GROUP'] == '2008 - 2010')]
    whole_training_data.reset_index(drop=True, inplace=True)
    patient_id_unique = whole_training_data['PATIENTID'].unique()
    split_point = np.linspace(0, len(patient_id_unique), cfg.folds + 1, dtype='int')
    whole_training_positive_rate = sum(whole_training_data[cfg.label_ch]) / len(whole_training_data)
    folds_min_mean_positive_var = 0.5
    best_random_state = cfg.great_random_state
    great_folds_training_data = None

    for random_state_num in range(10000, 20000, 1000):
        patient_id_after_shuffle = shuffle(patient_id_unique, random_state=random_state_num)
        patient_id_split_folds = [patient_id_after_shuffle[split_point[i]:split_point[i + 1]] for i in range(cfg.folds)]
        folds_training_data = [
            whole_training_data[whole_training_data['PATIENTID'].isin(patient_id_split_folds[i])] for i in range(cfg.folds)
        ]
        folds_positive_var = [
            abs(sum(folds_training_data[i][cfg.label_ch]) / len(folds_training_data[i]) - whole_training_positive_rate) ** 2
            for i in range(cfg.folds)
        ]
        folds_mean_positive_var = sum(folds_positive_var) / cfg.folds
        if folds_mean_positive_var < folds_min_mean_positive_var:
            best_random_state = random_state_num
            folds_min_mean_positive_var = folds_mean_positive_var
            great_folds_training_data = folds_training_data
            logger.info("Found better random seed %d, variance %.6f", best_random_state, folds_min_mean_positive_var)

    if great_folds_training_data is None:
        raise RuntimeError("Failed to generate a suitable cross-validation training set.")

    for i in range(cfg.folds):
        folds_training_index = list(range(cfg.folds))
        folds_validate_index = folds_training_index.pop(i)
        folds_training = pd.concat([great_folds_training_data[idx] for idx in folds_training_index])
        folds_training = shuffle(folds_training, random_state=best_random_state)
        folds_training.to_csv(
            cfg.folds_training_csv_path % (cfg.label, cfg.folds, best_random_state, i + 1),
            sep=";",
            index=False,
        )
        folds_validate = great_folds_training_data[folds_validate_index]
        folds_validate.to_csv(
            cfg.folds_validate_csv_path % (cfg.label, cfg.folds, best_random_state, i + 1),
            sep=";",
            index=False,
        )


def generate_common_testing_csv(cfg):
    whole_testing_data = pd.read_csv(cfg.whole_testing_csv_path, sep=';')
    # Filter out low-quality admission year groups.
    whole_testing_data = whole_testing_data[~(whole_testing_data['ADMISSION_YEAR_GROUP'] == '2008 - 2010')]
    whole_testing_data.reset_index(drop=True, inplace=True)
    whole_testing_data.to_csv(cfg.save_testing_csv_path, sep=";", index=False)
    logger.info("Common test set refreshed: %s", cfg.save_testing_csv_path)


def csv_to_data_frame(file_name: str, cfg) -> pd.DataFrame:
    df_data = pd.read_csv(file_name, sep=';')
    df_data[cfg.features_training] = df_data[cfg.features_training].astype(str)
    split_ch = np.full(len(df_data), ',')
    str_data = df_data[cfg.features_training[0]].astype(str)
    for i in range(1, len(cfg.features_training)):
        temp_lst = np.char.add(split_ch, df_data[cfg.features_training[i]].to_numpy())
        temp_lst[temp_lst == ',nan'] = ''
        str_data = np.char.add(str_data.to_numpy(), temp_lst)
    # Preserve standardized English keys when constructing the modeling frame.
    data_frame = pd.DataFrame({'STAYID': df_data['STAYID'], 'OBSERVATION': str_data.tolist(), cfg.label_ch: df_data[cfg.label_ch]})
    return data_frame


def generate_cross_data_pickle(cfg):
    test_data_frame = csv_to_data_frame(cfg.save_testing_csv_path, cfg)
    for i in range(cfg.folds):
        pickle_path = cfg.data_frame_pickle_pat % (cfg.label, cfg.folds, cfg.great_random_state, i + 1)
        if not os.path.exists(pickle_path):
            train_file = cfg.folds_training_csv_pa % (cfg.label, cfg.folds, cfg.great_random_state, i + 1)
            train_data_frame = csv_to_data_frame(train_file, cfg)
            pickle.dump((train_data_frame, test_data_frame), open(pickle_path, "wb"))
            logger.info("Generated DataFrame pickle: %s", pickle_path)
        else:
            logger.info("DataFrame pickle already exists: %s", pickle_path)

