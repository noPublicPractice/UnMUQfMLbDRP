import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
# plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']
plt.rcParams.update({'font.size': 44})


class const:
    # NOTE: Label identifiers use standardized English keys for consistency.
    label_ch_list = ['AF', 'AS', 'MI', 'CI']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    density = 200  # Sampling density
    interval_num = 20  # Number of groupby intervals
    statistical_coefficients_path = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-threshold-with-statistical-coefficients-dict-interval_num=%d.json'
    # model = 'ISP'
    model = 'SCF'


def plt_statistical_coefficients(classify_threshold_list, partial_positive_proportion_list, partial_negative_proportion_list):
    fig_3 = plt.figure(figsize=(5, 10))
    ax1 = fig_3.add_subplot(111)
    plt.plot(classify_threshold_list[:60], partial_negative_proportion_list[:60], label='Negative proportion per interval', c='k', linewidth=4)
    plt.plot(classify_threshold_list[60:], partial_positive_proportion_list[60:], label='Positive proportion per interval', c='r', linewidth=4)
    # plt.legend(loc=legend_loc, shadow=True, fancybox=True)
    plt.xlabel('Risk Score')
    if const.model == 'ISP':
        plt.ylabel('Interval Statistical Proportions')
    else:
        plt.ylabel('Statistical coefficient type\nConfidence Factor')
    # ax1.set_title(label_ch)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.spines['bottom'].set_linewidth(3)  # Set bottom axis line width
    ax1.spines['left'].set_linewidth(3)  # Set left axis line width
    ax1.spines['right'].set_linewidth(3)  # Set right axis line width
    ax1.spines['top'].set_linewidth(3)  # Set top axis line width
    fig_3.patch.set_alpha(0.0)
    plt.show()


class coefficient:
    temp_const = {"AF": (0.7327 - 0.2720) / (1.0 - 0.3), "AS": (0.8675 - 0.1424) / (1.0 - 0.595), "MI": (0.7895 - 0.2193) / (1.0 - 0.65), "CI": (0.8766 - 0.1312) / (1.0 - 0.7)}
    temp_coefficient_k = {"AF": ((0.95 / 0.9459) - 1) / (1 - 0.3), "AS": ((0.9 / 0.5167) - 1) / (1.0 - 0.595), "MI": ((0.95 / 0.8645) - 1) / (1.0 - 0.65), "CI": ((0.6 / 0.4375) - 1) / (1.0 - 0.7)}
    temp_coefficient_b = {"AF": 0.9981, "AS": -0.0898, "MI": 0.8163, "CI": 0.1333}


def calc(label_en, interval_right_endpoint, partial_proportion_dict, group_name):
    temp_k = coefficient.temp_coefficient_k[label_en] * interval_right_endpoint + coefficient.temp_coefficient_b[label_en]
    temp_b = coefficient.temp_const[label_en] * (1 - interval_right_endpoint)
    now_partial_proportion = temp_k * partial_proportion_dict['partial_positive_pro'][group_name] + temp_b
    return now_partial_proportion


def create_statistical_coefficients_list(label_en, label_ch):
    now_json_path_file_name = const.statistical_coefficients_path % (label_en, label_ch, const.interval_num)
    statistical_coefficients_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    threshold_interval = 1 / const.density  # Interval between classification thresholds
    classify_threshold_list = []
    partial_positive_proportion_list = []
    partial_negative_proportion_list = []
    for index, interval_right_endpoint in enumerate(np.linspace(threshold_interval, 1.0, const.density)):
        interval_left_endpoint = interval_right_endpoint - threshold_interval
        interval_middle_point = (interval_left_endpoint + interval_right_endpoint) / 2
        classify_threshold_list.append(interval_middle_point)
        group_name = '%.3f-%.3f' % (interval_left_endpoint, interval_right_endpoint)
        partial_negative_proportion_list.append(statistical_coefficients_dict['partial_negative_pro'][group_name])
        if const.model == 'ISP':
            partial_positive_proportion_list.append(statistical_coefficients_dict['partial_positive_pro'][group_name])
        else:
            now_partial_proportion = calc(label_en, interval_right_endpoint, statistical_coefficients_dict, group_name)
            partial_positive_proportion_list.append(now_partial_proportion)
    plt_statistical_coefficients(classify_threshold_list, partial_positive_proportion_list, partial_negative_proportion_list)


def do_work():
    for label_en, label_ch in zip(const.label_en_list, const.label_ch_list):
        # if label_en=='AF':
        create_statistical_coefficients_list(label_en, label_ch)


if __name__ == '__main__':
    do_work()


