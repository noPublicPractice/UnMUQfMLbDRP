from types import SimpleNamespace

"""
Centralize hyperparameters and constants for all run modes, organized by functional mode.
You can switch tasks by modifying run_mode or using the command-line argument --run_mode.
"""

run_mode = "af_data_prep"

# NOTE: Column names and domain labels now use standardized English keys.
LABEL_CH_LIST = ['AF', 'AS', 'MI', 'CI']
LABEL_EN_LIST = ['AF', 'AS', 'MI', 'CI']
ADAPTIVE_RANDOM_STATES = [15000, 13000, 12000, 15000]
COMMON_AU_ROC = {
    "AF": [0.881674, 0.881028, 0.881605, 0.882830, 0.881821, 0.881481],
    "AS": [0.867298, 0.866973, 0.863639, 0.865253, 0.868168, 0.863537],
    "MI": [0.920794, 0.920173, 0.920058, 0.920949, 0.919216, 0.919803],
    "CI": [0.842362, 0.839910, 0.839551, 0.843969, 0.839547, 0.839374],
}
PLOT_TEXT = {
    'en_primary': {
        'misclassified': 'Incorrect Classification',
        'correct': 'Correct Classification',
        'risk_score': 'Risk Score',
        'confidence_score': 'Confidence Score',
        'threshold': 'Threshold',
    },
    'en_alternate': {
        'misclassified': 'Incorrect Classification',
        'correct': 'Correct Classification',
        'risk_score': 'Risk Score (AUC Weight)',
        'confidence_score': 'Confidence Score (MCF×SCF)',
        'threshold': 'Threshold',
    },
}

af_data_prep_config = SimpleNamespace(
    patient_pkl_p='af_patient_patientid_stayid.pickle',
    icd_text_path='icd_lookup.csv',
    icd_patient_p='icd_patient_information.csv',
    data_pkl_path='data_.pickle',
    df_training_AF4_path='df_training_AF4.csv',
    random_state_num=12345,
)

cross_training_csv_config = SimpleNamespace(
    label='CI',
    label_ch='CI',
    folds=6,
    great_random_state=15000,
    whole_training_csv_path=r'disease_source_csv\df_training_%s.csv',
    folds_training_csv_path=r'cross_training_csv_pickle\%s_training_fold=%d-random_state=%d-exclude_fold=%d.csv',
    folds_validate_csv_path=r'cross_training_csv_pickle\%s_validation_fold=%d-random_state=%d-fold=%d.csv',
    whole_testing_csv_path=r'disease_source_csv\df_eval_common.csv',
    save_testing_csv_path=r'cross_training_csv_pickle\common_testing_set.csv',
)

cross_data_frame_config = SimpleNamespace(
    label='CI',
    label_ch='CI',
    folds=6,
    great_random_state=15000,
    save_testing_csv_path=r'cross_training_csv_pickle\common_testing_set.csv',
    folds_training_csv_pa=r'cross_training_csv_pickle\%s_training_fold=%d-random_state=%d-exclude_fold=%d.csv',
    data_frame_pickle_pat=r'cross_training_csv_pickle\%s_feature_label_space_fold=%d-random_state=%d-exclude_fold=%d.pickle',
    features_training=[
        'GENDER',
        'AGE_GROUP',
        'ADMISSION_TYPE',
        'DIAGNOSIS_HISTORY',
        'SURGERY_HISTORY',
        'LAB_RESULT',
        'ADMISSION_LOCATION',
        'INSURANCE',
        'LANGUAGE',
        'MARITAL_STATUS',
        'RACE',
    ],
)

train_model_config = SimpleNamespace(
    max_features_num=2000,
    label='CI',
    label_ch='CI',
    folds=6,
    great_random_state=15000,
    model_name='LGBM',
    data_frame_pickle_pat=r'cross_training_csv_pickle\%s_feature_label_space_fold=%d-random_state=%d-exclude_fold=%d.pickle',
    model_values_pkl_name=r'model_data_pkl\%s_model_fold=%d-random_state=%d-exclude_fold=%d-model_name=%s.pkl',
    evaluation_output='final_model_evaluation.csv',
)

threshold_metric_config = SimpleNamespace(
    label_ch_list=LABEL_CH_LIST,
    label_en_list=LABEL_EN_LIST,
    folds=6,
    great_random_state=ADAPTIVE_RANDOM_STATES,
    model_name='LGBM',
    model_values_joblib_name=r'..\step2_chapter3_chapter4_trained_models\risk_probability_matrix\model_data_pkl\%s_model_fold=%d-random_state=%d-exclude_fold=%d-model_name=%s.pkl',
    au_roc=COMMON_AU_ROC,
    density=200,
    do_write_json_path=r'..\plot_data_archive\%s%s-threshold-with-acc-ppv-tpr-dict-density=%d.json',
)

threshold_evaluation_config = SimpleNamespace(
    label_ch_list=LABEL_CH_LIST,
    label_en_list=LABEL_EN_LIST,
    folds=6,
    great_random_state=ADAPTIVE_RANDOM_STATES,
    model_name='LGBM',
    model_values_joblib_name=r'..\step2_chapter3_chapter4_trained_models\risk_probability_matrix\model_data_pkl\%s_model_fold=%d-random_state=%d-exclude_fold=%d-model_name=%s.pkl',
    au_roc=COMMON_AU_ROC,
    density=200,
    threshold_with_assessed_values_path=r'..\plot_data_archive\%s%s-threshold-with-acc-ppv-tpr-dict-density=%d.json',
)

stat_coefficient_config = SimpleNamespace(
    label_ch_list=LABEL_CH_LIST,
    label_en_list=LABEL_EN_LIST,
    folds=6,
    great_random_state=ADAPTIVE_RANDOM_STATES,
    model_name='LGBM',
    model_values_joblib_name=r'..\step2_chapter3_chapter4_trained_models\risk_probability_matrix\model_data_pkl\%s_model_fold=%d-random_state=%d-exclude_fold=%d-model_name=%s.pkl',
    au_roc=COMMON_AU_ROC,
    density=200,
    interval_num=20,
    do_write_json_path=r'..\plot_data_archive\%s%s-threshold-with-statistical-coefficients-dict-interval_num=%d.json',
)

confidence_artifact_config = SimpleNamespace(
    label_ch_list=LABEL_CH_LIST,
    label_en_list=LABEL_EN_LIST,
    density=200,
    interval_num=20,
    threshold_with_assessed_values_path=r'..\plot_data_archive\%s%s-threshold-with-acc-ppv-tpr-dict-density=%d.json',
    statistical_coefficients_path=r'..\plot_data_archive\%s%s-threshold-with-statistical-coefficients-dict-interval_num=%d.json',
    folds=6,
    great_random_state=ADAPTIVE_RANDOM_STATES,
    model_name='LGBM',
    model_values_joblib_name=r'..\step2_chapter3_chapter4_trained_models\risk_probability_matrix\model_data_pkl\%s_model_fold=%d-random_state=%d-exclude_fold=%d-model_name=%s.pkl',
    au_roc=COMMON_AU_ROC,
    confidence_score_f='MCF×SCF',
    do_write_confidence_score_json_path=r'..\plot_data_archive\%s%s-confidence-score-dict-func=%s.json',
    do_write_predict_information_json_path=r'..\plot_data_archive\%s%s-predict-information-dict-func=%s.json',
    temp_const={"AF": (0.7327 - 0.2720) / (1.0 - 0.3), "AS": (0.8675 - 0.1424) / (1.0 - 0.595),
                "MI": (0.7895 - 0.2193) / (1.0 - 0.65), "CI": (0.8766 - 0.1312) / (1.0 - 0.7)},
    temp_coefficient_k={"AF": ((0.95 / 0.9459) - 1) / (1 - 0.3), "AS": ((0.9 / 0.5167) - 1) / (1.0 - 0.595),
                        "MI": ((0.95 / 0.8645) - 1) / (1.0 - 0.65), "CI": ((0.6 / 0.4375) - 1) / (1.0 - 0.7)},
    temp_coefficient_b={"AF": 0.9981, "AS": -0.0898, "MI": 0.8163, "CI": 0.1333},
)

MODE_CONFIGS = {
    "af_data_prep": af_data_prep_config,
    "generate_cross_training_csv": cross_training_csv_config,
    "generate_common_testing_csv": cross_training_csv_config,
    "generate_cross_data_pickle": cross_data_frame_config,
    "train_models": train_model_config,
    "reevaluate_models": train_model_config,
    "generate_threshold_metrics": threshold_metric_config,
    "print_threshold_evaluation": threshold_evaluation_config,
    "compute_statistical_coefficients": stat_coefficient_config,
    "prepare_confidence_artifacts": confidence_artifact_config,
}

AVAILABLE_MODES = tuple(MODE_CONFIGS.keys())

