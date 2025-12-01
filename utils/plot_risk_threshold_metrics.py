import json
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
# plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']


class const:
    # NOTE: Label identifiers use standardized English keys for consistency.
    label_ch_list = ['AF', 'AS', 'MI', 'CI']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    density = 200
    threshold_with_assessed_values_path = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-threshold-with-acc-ppv-tpr-dict-density=%d.json'


def plt_assessed_values(label_en, label_ch):
    now_json_path_file_name = const.threshold_with_assessed_values_path % (label_en, label_ch, const.density)
    assessed_values_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    fig_3 = plt.figure(figsize=(16, 8))
    ax1 = fig_3.add_subplot(111)
    plt.plot(assessed_values_dict['classify_threshold'], assessed_values_dict['acc'], label='accuracy', c='k', linewidth=4)
    plt.plot(assessed_values_dict['classify_threshold'], assessed_values_dict['ppv'], label='precision', c='r', linewidth=4)
    plt.plot(assessed_values_dict['classify_threshold'], assessed_values_dict['tpr'], label='sensitivity', c='g', linewidth=4)
    plt.plot(assessed_values_dict['classify_threshold'], assessed_values_dict['f1'], label='F1-score', c='b', linewidth=4)
    plt.plot(assessed_values_dict['classify_threshold'], assessed_values_dict['acc_multiply_tpr'], label='accuracy×sensitivity', c='y', linewidth=4)
    plt.rcParams.update({'font.size': 48})
    plt.legend(shadow=True, fancybox=True, bbox_to_anchor=(1.05, 0.3), loc=3, borderaxespad=0)
    plt.xlabel('Risk Probability Binary Classification Threshold')
    plt.ylabel('Model Performance\nMetric Value')
    # ax1.set_title(label_ch)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.spines['bottom'].set_linewidth(3)  # Set bottom axis line width
    ax1.spines['left'].set_linewidth(3)  # Set left axis line width
    ax1.spines['right'].set_linewidth(3)  # Set right axis line width
    ax1.spines['top'].set_linewidth(3)  # Set top axis line width
    plt.show()


plt_assessed_values(const.label_en_list[0], const.label_ch_list[0])


def do_work():
    for label_en, label_ch in zip(const.label_en_list, const.label_ch_list):
        plt_assessed_values(label_en, label_ch)


if __name__ == '__main__':
    do_work()


