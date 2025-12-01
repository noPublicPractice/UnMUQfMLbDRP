import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = {
    'Without division of\nhigh- and low-confidence regions': np.array([(3962, 2408), (3875, 36115)]),
    '1st Order Threshold\nHigh Confidence Level': np.array([(2489, 1277), (1929, 26587)]),
    '1st order Threshold\nLow Confidence Level': np.array([(1473, 1131), (1946, 9528)]),
    '2nd order Threshold\nHigh Confidence Level': np.array([(2452, 1480), (2317, 24081)]),
    '2nd order Threshold\nLow Confidence Level': np.array([(1510, 928), (1558, 12034)])
}
fig, axs = plt.subplots(1, 5, figsize=(19, 3))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    pcm = axs[index].imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=axs[index])

    classes_x = ['Positive', 'Negative']
    classes_y = ['Positive', 'Negative']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        axs[index].text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    # plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)  # Place xticks on top
    axs[index].set_xticks(np.arange(len(classes_x)), classes_x)
    axs[index].set_yticks(np.arange(len(classes_y)), classes_y)
    axs[index].set_title(now_kernel)  # Update title
    axs[index].set_xlabel('Prediction')
    if now_kernel == list(confusion_matrix.keys())[0]:
        axs[index].set_ylabel('Labels', rotation=0)
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    '1st Order Threshold\nHigh Confidence Level': np.array([(2489, 1277), (1929, 26587)]),
    '1st order Threshold\nLow Confidence Level': np.array([(1473, 1131), (1946, 9528)]),
    '2nd order Threshold\nHigh Confidence Level': np.array([(2452, 1480), (2317, 24081)]),
    '2nd order Threshold\nLow Confidence Level': np.array([(1510, 928), (1558, 12034)])
}
fig, axs = plt.subplots(1, 4, figsize=(15, 3))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    pcm = axs[index].imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=axs[index])

    classes_x = ['Positive', 'Negative']
    classes_y = ['Positive', 'Negative']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        axs[index].text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    # plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)  # Place xticks on top
    axs[index].set_xticks(np.arange(len(classes_x)), classes_x)
    axs[index].set_yticks(np.arange(len(classes_y)), classes_y)
    axs[index].set_title(now_kernel)  # Update title
    axs[index].set_xlabel('Prediction')
    if now_kernel == list(confusion_matrix.keys())[0]:
        axs[index].set_ylabel('Labels', rotation=0)
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    '1st Order Threshold\nHigh Confidence Level': np.array([(2489, 1277), (1929, 26587)]),
    '1st order Threshold\nLow Confidence Level': np.array([(1473, 1131), (1946, 9528)]),
    '2nd order Threshold\nHigh Confidence Level': np.array([(2452, 1480), (2317, 24081)]),
    '2nd order Threshold\nLow Confidence Level': np.array([(1510, 928), (1558, 12034)])
}
subplots_index = {
    list(confusion_matrix.keys())[0]: [0, 0],
    list(confusion_matrix.keys())[1]: [0, 1],
    list(confusion_matrix.keys())[2]: [1, 0],
    list(confusion_matrix.keys())[3]: [1, 1]
}
fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    ax = axs[subplots_index[now_kernel][0], subplots_index[now_kernel][1]]
    pcm = ax.imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=ax)

    classes_x = ['Positive', 'Negative']
    classes_y = ['Positive', 'Negative']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        ax.text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    ax.set_xticks(np.arange(len(classes_x)), classes_x)
    ax.set_yticks(np.arange(len(classes_y)), classes_y)
    ax.set_title(now_kernel)  # Update title
    if '2nd' in now_kernel:
        ax.set_xlabel('Prediction')
    if 'High' in now_kernel:
        ax.set_ylabel('Labels', rotation=0)
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = np.array([
    (2489, 1277, 1473, 1131, 2452, 1480, 1510, 928),
    (1929, 26587, 1946, 9528, 2317, 24081, 1558, 12034)
])

fig_3 = plt.figure(figsize=(8, 4))
plt.rcParams.update({'font.size': 12})

plt.imshow(confusion_matrix, interpolation='nearest', cmap='summer')  # Display matrix as pixels
plt.colorbar(orientation='horizontal')

classes_x = ['Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative', 'Positive', 'Negative']
classes_y = ['Positive', 'Negative']
for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix.size, 2)):  # Reshape removes one tensor dimension
    plt.text(j, i, confusion_matrix[i, j], va='center', ha='center')  # Iterate matrix indices and display numbers

plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)  # Place xticks on top
plt.xticks(np.arange(len(classes_x)), classes_x)
plt.yticks(np.arange(len(classes_y)), classes_y)
plt.ylabel('Labels', rotation=0)
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

confusion_matrix = {
    'TP': np.array([(2489, 2452), (1473, 1510)]),
    'FN': np.array([(1277, 1480), (1131, 928)]),
    'FP': np.array([(1929, 2317), (1946, 1558)]),
    'TN': np.array([(26587, 24081), (9528, 12034)])
}
fig, axs = plt.subplots(1, 4, figsize=(15, 3))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    pcm = axs[index].imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=axs[index])

    classes_x = ['1st order', '2nd order']
    classes_y = ['High', 'Low']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        axs[index].text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    axs[index].set_xticks(np.arange(len(classes_x)), classes_x)
    axs[index].set_yticks(np.arange(len(classes_y)), classes_y)
    axs[index].set_title(now_kernel)  # Update title
    axs[index].set_xlabel('Threshold Category')
    if now_kernel == list(confusion_matrix.keys())[0]:
        axs[index].set_ylabel('Confidence Level')
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    'TP': np.array([(2489, 2452), (1473, 1510)]),
    'FN': np.array([(1277, 1480), (1131, 928)]),
    'FP': np.array([(1929, 2317), (1946, 1558)]),
    'TN': np.array([(26587, 24081), (9528, 12034)])
}
subplots_index = {
    list(confusion_matrix.keys())[0]: [0, 0],
    list(confusion_matrix.keys())[1]: [0, 1],
    list(confusion_matrix.keys())[2]: [1, 0],
    list(confusion_matrix.keys())[3]: [1, 1]
}
fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    ax = axs[subplots_index[now_kernel][0], subplots_index[now_kernel][1]]
    pcm = ax.imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=ax)

    classes_x = ['1st order', '2nd order']
    classes_y = ['High', 'Low']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        ax.text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    ax.set_xticks(np.arange(len(classes_x)), classes_x)
    ax.set_yticks(np.arange(len(classes_y)), classes_y)
    ax.set_title(now_kernel)  # Update title
    if now_kernel in ['TN', 'FP']:
        ax.set_xlabel('Threshold Category')
    if now_kernel in ['TP', 'FP']:
        ax.set_ylabel('Confidence Level')
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart


