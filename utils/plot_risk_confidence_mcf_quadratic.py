import json
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
# plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']


class const:
    # NOTE: Label identifiers use standardized English keys for consistency.
    label_ch_list = ['AF', 'AS', 'MI', 'CI']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    confidence_score_f = 'MCF'
    # Confidence score data and prediction evaluation sources
    confidence_score_dict = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-confidence-score-dict-func=%s.json'
    do_write_predict_information_json_path = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-predict-information-dict-func=%s.json'
    # Curve-fitting parameters
    density = 200
    popt = {'AF': [], 'AS': [], 'MI': [], 'CI': []}
    # Plot text resources
    lan = 'en'
    pic_str = {
        'en': {
            'misclassified': 'Incorrect Classification',
            'correct': 'Correct Classification',
            'threshold': 'Threshold',
            'risk_score': 'Risk Score',
            'confidence_score': 'Confidence Score',
        }
    }


##### Fit decision boundary, store fitted threshold parameters, and plot boundary
def fit_func(x, a, b, c):  # The variable must be provided as the first argument
    return a * x ** 2 + b * x + c


def curve_fit_boundary(label_en, confidence_score_dict, boundary_x_coordinate):
    # Risk probability
    pred_pro_weighted_sum = confidence_score_dict["true"]["pred_pro_weighted_sum"] + confidence_score_dict["false"]["pred_pro_weighted_sum"]
    # Confidence score
    pred_pro_confidence = confidence_score_dict["true"]["pred_pro_confidence"] + confidence_score_dict["false"]["pred_pro_confidence"]
    const.popt[label_en], pcov = opt.curve_fit(fit_func, pred_pro_weighted_sum, pred_pro_confidence)
    boundary_y_coordinate_fit = fit_func(boundary_x_coordinate, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2])
    return boundary_y_coordinate_fit


def draw_boundary(confidence_score_dict, boundary_x_coordinate, boundary_y_coordinate_fit):  # Plot the decision boundary using discrete points
    fig_3 = plt.figure(figsize=(16, 8))
    ax1 = fig_3.add_subplot(111)
    plt.scatter(
        confidence_score_dict["false"]["pred_pro_weighted_sum"],
        confidence_score_dict["false"]["pred_pro_confidence"],
        label=const.pic_str[const.lan]['misclassified'],
        c='r',
        s=1,
    )
    plt.scatter(
        confidence_score_dict["true"]["pred_pro_weighted_sum"],
        confidence_score_dict["true"]["pred_pro_confidence"],
        label=const.pic_str[const.lan]['correct'],
        c='k',
        s=1,
    )
    plt.plot(boundary_x_coordinate, boundary_y_coordinate_fit, label=const.pic_str[const.lan]['threshold'], c='b', linewidth=3)
    plt.legend(loc="lower right", shadow=True, fancybox=True, markerscale=24)  # "markerscale=24": adjust marker size within the legend only
    plt.rcParams.update({'font.size': 48})
    plt.xlabel(const.pic_str[const.lan]['risk_score'])
    plt.ylabel(const.pic_str[const.lan]['confidence_score'])
    plt.title('')
    ax1.set_xlim([-0.02, 1.02])
    # ax1.set_ylim([0.75, 1.02])
    ax1.spines['bottom'].set_linewidth(3)  # Set bottom axis line width
    ax1.spines['left'].set_linewidth(3)  # Set left axis line width
    ax1.spines['right'].set_linewidth(3)  # Set right axis line width
    ax1.spines['top'].set_linewidth(3)  # Set top axis line width
    plt.show()


##### Count labels and compute evaluation metrics
def prepare_predict_information(label_en, label_ch):
    now_json_path_file_name = const.do_write_predict_information_json_path % (label_en, label_ch, const.confidence_score_f)
    predict_information_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    pred_pro_weighted_sum = np.array(predict_information_dict["pred_pro_weighted_sum"])
    pred_pro_confidence_score = np.array(predict_information_dict["pred_pro_confidence_score"])
    # JSON cannot persist np.int, so floats are cast back to int here.
    test_data_label_arr = np.array(predict_information_dict["test_data_label_arr"]).astype(int)
    pred_label_arr = np.array(predict_information_dict["pred_label_arr"]).astype(int)
    return pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr


def prepare_predict_information_dict(label_en, pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr):
    judge_y = pred_pro_confidence_score - fit_func(pred_pro_weighted_sum, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2])
    # When popt retains fewer decimals, the counts in up/down arrays may differ slightly by a few samples.
    above_fit_func_index = np.where(judge_y >= 0)[0]
    below_fit_func_index = np.where(judge_y < 0)[0]
    above_fit_func_predict_information = {
        "pred_pro_weighted_sum": pred_pro_weighted_sum[above_fit_func_index],
        "test_data_label_arr": test_data_label_arr[above_fit_func_index],
        "pred_label_arr": pred_label_arr[above_fit_func_index],
    }
    below_fit_func_predict_information = {
        "pred_pro_weighted_sum": pred_pro_weighted_sum[below_fit_func_index],
        "test_data_label_arr": test_data_label_arr[below_fit_func_index],
        "pred_label_arr": pred_label_arr[below_fit_func_index],
    }
    return above_fit_func_predict_information, below_fit_func_predict_information


def calc_accuracy(tp, tn, amount_sum):
    return (tp + tn) / amount_sum


def calc_ppv(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def calc_tpr(tp, fn):
    return tp / (tp + fn)


def calc_f1(tp, fn, fp):
    return 2 * tp / (2 * tp + fn + fp)  # Equivalent to 2 * ppv * tpr / (ppv + tpr)


def count_label_arr_and_calc_assessed_values(predict_information_dict):
    predict_positive_label_index = np.where(predict_information_dict["pred_label_arr"] == 1)[0]
    predict_negative_label_index = np.where(predict_information_dict["pred_label_arr"] == 0)[0]
    test_data_positive_label_index = np.where(predict_information_dict["test_data_label_arr"] == 1)[0]
    test_data_negative_label_index = np.where(predict_information_dict["test_data_label_arr"] == 0)[0]
    # Intersections identify TP/TN/FP/FN collections
    tp_index = list(set(predict_positive_label_index) & set(test_data_positive_label_index))
    fn_index = list(set(predict_negative_label_index) & set(test_data_positive_label_index))
    fp_index = list(set(predict_positive_label_index) & set(test_data_negative_label_index))
    tn_index = list(set(predict_negative_label_index) & set(test_data_negative_label_index))
    tp_num = len(tp_index)
    fn_num = len(fn_index)
    fp_num = len(fp_index)
    tn_num = len(tn_index)
    amount_sum = tp_num + fn_num + fp_num + tn_num
    false_positive_rate, true_positive_rate, thresholds_list = roc_curve(
        predict_information_dict["test_data_label_arr"],
        predict_information_dict["pred_pro_weighted_sum"],
        pos_label=1,
    )
    calc_assessed_values = {
        "acc": calc_accuracy(tp_num, tn_num, amount_sum),
        "ppv": calc_ppv(tp_num, fp_num),
        "tpr": calc_tpr(tp_num, fn_num),
        "f1": calc_f1(tp_num, fn_num, fp_num),
        "au_roc": auc(false_positive_rate, true_positive_rate),
        "tp_num": tp_num,
        "fn_num": fn_num,
        "fp_num": fp_num,
        "tn_num": tn_num,
    }
    return calc_assessed_values


def print_assessed_values(label_en, label_ch):
    float_assessed_values_name_list = ["ACC", "PPV", "TPR", "F1", "AUC"]
    float_assessed_values_name_lowercase_list = ["acc", "ppv", "tpr", "f1", "au_roc"]
    int_assessed_values_name_list = ["TP", "FN", "FP", "TN"]
    int_assessed_values_name_lowercase_list = ["tp_num", "fn_num", "fp_num", "tn_num"]
    # Load risk probability, confidence score, label space, and prediction outcomes
    pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr = prepare_predict_information(label_en, label_ch)
    # Separate high/low confidence regions for risk probability, confidence score, label space, and predictions
    above_fit_func_predict_information, below_fit_func_predict_information = prepare_predict_information_dict(
        label_en,
        pred_pro_weighted_sum,
        pred_pro_confidence_score,
        test_data_label_arr,
        pred_label_arr,
    )
    # Evaluate high-confidence region predictions
    calc_assessed_values_above_fit_func = count_label_arr_and_calc_assessed_values(above_fit_func_predict_information)
    # Evaluate low-confidence region predictions
    calc_assessed_values_below_fit_func = count_label_arr_and_calc_assessed_values(below_fit_func_predict_information)
    # Print fitted threshold curve
    print('%s fitted threshold curve: y = %.3f x^2 + %.3f x + %.3f\n' % (label_ch, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2]))
    # Output evaluation summary
    print("%s\n\tHigh confidence interval\tLow confidence interval" % label_ch)
    for name, name_lowercase in zip(float_assessed_values_name_list, float_assessed_values_name_lowercase_list):
        print("%s\t%.3f\t\t%.3f" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))
    for name, name_lowercase in zip(int_assessed_values_name_list, int_assessed_values_name_lowercase_list):
        print("%s\t%d\t\t%d" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))


def do_work(label_en, label_ch):
    # Prepare sample and label spaces required for logistic regression
    now_json_path_file_name = const.confidence_score_dict % (label_en, label_ch, const.confidence_score_f)
    confidence_score_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    # Fit decision boundary and store fitted threshold curve parameters
    boundary_y_coordinate_fit = curve_fit_boundary(label_en, confidence_score_dict, boundary_x_coordinate := np.linspace(0, 1, const.density))
    # Plot risk probability vs. confidence score, decision boundary points, and fitted curve
    draw_boundary(confidence_score_dict, boundary_x_coordinate, boundary_y_coordinate_fit)
    # Output fitted threshold curve and evaluation metrics
    print_assessed_values(label_en, label_ch)


# label_en, label_ch = const.label_en_list[0],const.label_ch_list[0]
# confidence_score_dict = prepare_data(label_en, label_ch)
# boundary_y_coordinate_fit = curve_fit_boundary(label_en, confidence_score_dict, boundary_x_coordinate := np.linspace(0, 1, const.density))
# draw_boundary(confidence_score_dict, boundary_x_coordinate, boundary_y_coordinate_fit)
# print_assessed_values(label_en, label_ch)
do_work(const.label_en_list[0], const.label_ch_list[0])
do_work(const.label_en_list[0], const.label_ch_list[0])
do_work(const.label_en_list[1], const.label_ch_list[1])
do_work(const.label_en_list[2], const.label_ch_list[2])
do_work(const.label_en_list[3], const.label_ch_list[3])


