import json
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
# plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']


class const:
    # NOTE: Label identifiers use standardized English keys for consistency.
    label_ch_list = ['AF', 'AS', 'MI', 'CI']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    confidence_score_f = 'MCF×SCF'
    # Confidence score data and prediction evaluation sources
    confidence_score_dict = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-confidence-score-dict-func=%s.json'
    do_write_predict_information_json_path = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-predict-information-dict-func=%s.json'
    # Logistic regression parameters
    degree = 6
    lam = 100
    density = 200
    threshhold = 2 * 10 ** -2
    # Curve-fitting parameters
    popt = {}
    legend_loc = {'AF': 'lower right', 'AS': 'lower right', 'MI': 'lower right', 'CI': 'lower right'}
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


##### Prepare sample space and label space for logistic regression, plus related processing
def prepare_data(label_en, label_ch):
    now_json_path_file_name = const.confidence_score_dict % (label_en, label_ch, const.confidence_score_f)
    confidence_score_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    # Positive samples
    pred_pro_weighted_sum_true = confidence_score_dict["true"]["pred_pro_weighted_sum"]
    pred_pro_confidence_true = confidence_score_dict["true"]["pred_pro_confidence"]
    # Negative samples
    pred_pro_weighted_sum_false = confidence_score_dict["false"]["pred_pro_weighted_sum"]
    pred_pro_confidence_false = confidence_score_dict["false"]["pred_pro_confidence"]
    logistic_inputs_x = np.array([pred_pro_weighted_sum_true + pred_pro_weighted_sum_false, pred_pro_confidence_true + pred_pro_confidence_false])
    logistic_inputs_y = np.concatenate((np.ones(len(pred_pro_weighted_sum_true)), np.zeros(len(pred_pro_weighted_sum_false)))).astype(int)
    return logistic_inputs_x.T, logistic_inputs_y


def feature_mapping(x1, x2, as_ndarray=False):
    data = {}
    for i in np.arange(const.degree + 1):
        for p in np.arange(i + 1):
            data["f%d%d" % (i - p, p)] = np.power(x1, i - p) * np.power(x2, p)
    return np.array(pd.DataFrame(data)) if as_ndarray else pd.DataFrame(data)


##### Logistic regression: fit parameters
def sigmoid(z):  # Sigmoid function
    return 1 / (1 + np.exp(-z))


def calc_regularized_loss(theta, x, y):  # Regularized loss function
    classify_predict = sigmoid(np.dot(x, theta))
    training_loss = np.mean(-y * np.log(classify_predict) - (1 - y) * np.log(1 - classify_predict))
    parameters_loss = (1 / (2 * len(x))) * np.power(theta[1:], 2).sum()  # Regularization term to reduce overfitting
    return training_loss + parameters_loss


def calc_regularized_gradient(theta, x, y):  # Regularized gradient function
    classify_predict = sigmoid(np.dot(x, theta))
    training_gradient = (1 / len(x)) * np.dot(x.T, classify_predict - y)
    parameters_gradient_higher_term = (const.lam / len(x)) * theta[1:]
    parameters_gradient = np.concatenate([np.array([0]), parameters_gradient_higher_term])
    return training_gradient + parameters_gradient


def feature_mapped_logistic_regression(logistic_inputs_x, logistic_inputs_y):
    inputs_x = feature_mapping(logistic_inputs_x[:, 0], logistic_inputs_x[:, 1], as_ndarray=True)
    theta = np.zeros(inputs_x.shape[1])
    res = opt.minimize(fun=calc_regularized_loss, x0=theta, args=(inputs_x, logistic_inputs_y), method='TNC', jac=calc_regularized_gradient)
    final_theta = res.x
    return final_theta


##### Locate discrete decision boundary points, fit the boundary, store threshold parameters, and plot the boundary
def find_decision_boundary(theta):  # Locate discrete points along the decision boundary
    x_interval_value = np.linspace(0, 1, const.density)
    y_interval_value = np.linspace(0, 1, const.density)
    coordinate_value = [(x, y) for x in x_interval_value for y in y_interval_value]  # Generate grid coordinates
    x_coordinate, y_coordinate = zip(*coordinate_value)
    map_high_dimensional = feature_mapping(x_coordinate, y_coordinate)  # Rows: density^2, columns: (degree+1)(degree+2)/2
    distance = np.dot(np.array(map_high_dimensional), theta)
    decision = map_high_dimensional[np.abs(distance) < const.threshhold]
    # Notes:
    # - "*coordinate_value" converts the list into tuples of coordinates.
    # - Mapping 2D features to high-dimensional space results in (degree+1)(degree+2)/2 dimensions.
    # - Each theta coefficient multiplies the corresponding high-dimensional feature to form a prediction value,
    #   which can be regarded as the distance between a point and the decision boundary.
    # - Points whose distance absolute value is below threshhold=0.002 are treated as lying on the boundary.
    # - decision.f10 and decision.f01 correspond to the x and y coordinates (x^1*y^0 and x^0*y^1 respectively).
    return decision.f10, decision.f01


def fit_func_1(x, a, b, c):  # Variable must be first argument
    return a * x ** 2 + (b + c) * x


def fit_func_2(x, a, b, c):  # Variable must be first argument
    return a * (x - b) ** 2 + c


def fit_func(x, a, b, c):  # Variable must be first argument
    return a * x ** 2 + b * x + c


def curve_fit_boundary(label_en, boundary_x_coordinate, boundary_y_coordinate):
    const.popt[label_en], pcov = opt.curve_fit(fit_func, boundary_x_coordinate, boundary_y_coordinate)
    boundary_y_coordinate_fit = fit_func(boundary_x_coordinate, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2])
    return boundary_y_coordinate_fit


def draw_boundary(label_en, logistic_inputs_x, logistic_inputs_y, boundary_x_coordinate, boundary_y_coordinate_fit, boundary_y_coordinate):
    fig_3 = plt.figure(figsize=(16, 8))
    ax1 = fig_3.add_subplot(111)
    plt.scatter(
        logistic_inputs_x[:, 0][logistic_inputs_y == 0],
        logistic_inputs_x[:, 1][logistic_inputs_y == 0],
        label=const.pic_str[const.lan]['misclassified'],
        c='r',
        s=1,
    )
    plt.scatter(
        logistic_inputs_x[:, 0][logistic_inputs_y == 1],
        logistic_inputs_x[:, 1][logistic_inputs_y == 1],
        label=const.pic_str[const.lan]['correct'],
        c='k',
        s=1,
    )  # Plot samples by class
    # plt.scatter(boundary_x_coordinate, boundary_y_coordinate, label='Fitted decision boundary', c='g', s=5)
    plt.plot(boundary_x_coordinate, boundary_y_coordinate_fit, label=const.pic_str[const.lan]['threshold'], c='b', linewidth=3)
    plt.legend(loc=const.legend_loc[label_en], shadow=True, fancybox=True, markerscale=24)  # "markerscale=24": adjust legend marker size only
    plt.rcParams.update({'font.size': 44})
    plt.xlabel(const.pic_str[const.lan]['risk_score'])
    plt.ylabel(const.pic_str[const.lan]['confidence_score'])
    # plt.title('')
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
    print('%s fitted threshold curve: y = %.3f (x-%.3f)^2 + %.3f\n' % (label_ch, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2]))
    # Output evaluation summary
    print("%s\n\tHigh confidence interval\tLow confidence interval" % label_ch)
    for name, name_lowercase in zip(float_assessed_values_name_list, float_assessed_values_name_lowercase_list):
        print("%s\t%.3f\t\t%.3f" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))
    for name, name_lowercase in zip(int_assessed_values_name_list, int_assessed_values_name_lowercase_list):
        print("%s\t%d\t\t%d" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))


def do_work(label_en, label_ch):
    # Prepare logistic regression sample and label spaces
    logistic_inputs_x, logistic_inputs_y = prepare_data(label_en, label_ch)
    # Fit logistic regression parameters
    final_theta = feature_mapped_logistic_regression(logistic_inputs_x, logistic_inputs_y)
    # Locate discrete decision boundary points
    boundary_x_coordinate, boundary_y_coordinate = find_decision_boundary(final_theta)
    # Fit the boundary and store threshold parameters
    boundary_y_coordinate_fit = curve_fit_boundary(label_en, boundary_x_coordinate, boundary_y_coordinate)
    # Plot risk score vs. confidence score, boundary points, and fitted curve
    draw_boundary(label_en, logistic_inputs_x, logistic_inputs_y, boundary_x_coordinate, boundary_y_coordinate, boundary_y_coordinate_fit)
    # Output fitted threshold curve and evaluation metrics
    print_assessed_values(label_en, label_ch)

do_work(const.label_en_list[0], const.label_ch_list[0])
do_work(const.label_en_list[1], const.label_ch_list[1])
do_work(const.label_en_list[2], const.label_ch_list[2])
do_work(const.label_en_list[3], const.label_ch_list[3])


