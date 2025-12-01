import json
import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
# plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']


class const:
    # NOTE: Label identifiers use standardized English keys for consistency.
    label_ch_list = ['AF', 'AS', 'MI', 'CI']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    confidence_score_f = 'MCF'
    confidence_score_dict = r'..\..\PycharmProject-202412-ensemble-learning\plot_data_archive\%s%s-confidence-score-dict-func=%s.json'
    lan = 'en'
    pic_str = {
        'en': {
            'misclassified': 'Incorrect Classification',
            'correct': 'Correct Classification',
            'risk_score': 'Weight Risk Score',
            'confidence_score': 'Confidence Score',
        }
    }


def plt_confidence(label_en, label_ch):
    confidence_score_dict = json.load(open(const.confidence_score_dict % (label_en, label_ch, const.confidence_score_f), 'r', encoding='utf-8'))
    fig_3 = plt.figure(figsize=(8, 8))
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
    # plt.scatter(confidence_score_dict["false"]["pred_pro_weighted_sum"], confidence_score_dict["false"]["pred_pro_confidence"], label='Incorrect Classification', c='r', s=1)
    # x = np.linspace(-0.02, 1.02, 100)
    # y = 0.08*(np.exp(np.negative(20*((x+np.negative(0.5))**2)))+11.2)
    # y = 0.08*(np.exp(np.negative(20*((x+np.negative(0.62))**2)))+11.4)
    # plt.plot(x, y, label='Threshold curve', c='green', linewidth=4)  # lightblue
    plt.rcParams.update({'font.size': 48})
    # plt.legend(loc="lower right", shadow=True, fancybox=True)
    plt.xlabel(const.pic_str[const.lan]['risk_score'])
    plt.ylabel(const.pic_str[const.lan]['confidence_score'])
    ax1.set_title("MCF")
    # ax1.set_xlim([0.3, 1.02])
    # ax1.set_ylim([0.15, 1.05])
    ax1.spines['bottom'].set_linewidth(3)  # Set bottom axis line width
    ax1.spines['left'].set_linewidth(3)  # Set left axis line width
    ax1.spines['right'].set_linewidth(3)  # Set right axis line width
    ax1.spines['top'].set_linewidth(3)  # Set top axis line width
    plt.show()


plt_confidence(const.label_en_list[0], const.label_ch_list[0])


def do_work():
    for label_en, label_ch in zip(const.label_en_list, const.label_ch_list):
        plt_confidence(label_en, label_ch)


if __name__ == '__main__':
    do_work()


