import numpy as np
import matplotlib.pyplot as plt

confusion_matrix = {
    'AF\nHigh Confidence Level': np.array([(3271, 1577), (2338, 33075)]),
    'AS\nHigh Confidence Level': np.array([(198, 115), (418, 31735)]),
    'MI\nHigh Confidence Level': np.array([(557, 336), (467, 39827)]),
    'CI\nHigh Confidence Level': np.array([(139, 117), (208, 30061)]),
    'AF\nLow Confidence Level': np.array([(691, 831), (1537, 3040)]),
    'AS\nLow Confidence Level': np.array([(27, 440), (111, 13316)]),
    'MI\nLow Confidence Level': np.array([(131, 464), (375, 4203)]),
    'CI\nLow Confidence Level': np.array([(39, 546), (176, 15074)])
}
subplots_index = {
    list(confusion_matrix.keys())[0]: [0, 0],
    list(confusion_matrix.keys())[1]: [0, 1],
    list(confusion_matrix.keys())[2]: [0, 2],
    list(confusion_matrix.keys())[3]: [0, 3],
    list(confusion_matrix.keys())[4]: [1, 0],
    list(confusion_matrix.keys())[5]: [1, 1],
    list(confusion_matrix.keys())[6]: [1, 2],
    list(confusion_matrix.keys())[7]: [1, 3]
}
fig, axs = plt.subplots(2, 4, figsize=(15, 6))  # Create subplots with a specific size
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
    ax.set_title(now_kernel)
    if 'Low' in now_kernel:
        ax.set_xlabel('Prediction')
    if 'AF' in now_kernel:
        ax.set_ylabel('Labels', rotation=0)
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    'AF\nHigh Confidence Level': np.array([(3271, 1577), (2338, 33075)]),
    'AF\nLow Confidence Level': np.array([(691, 831), (1537, 3040)]),
    'AS\nHigh Confidence Level': np.array([(198, 115), (418, 31735)]),
    'AS\nLow Confidence Level': np.array([(27, 440), (111, 13316)]),
    'MI\nHigh Confidence Level': np.array([(557, 336), (467, 39827)]),
    'MI\nLow Confidence Level': np.array([(131, 464), (375, 4203)]),
    'CI\nHigh Confidence Level': np.array([(139, 117), (208, 30061)]),
    'CI\nLow Confidence Level': np.array([(39, 546), (176, 15074)])
}
subplots_index = {
    list(confusion_matrix.keys())[0]: [0, 0],
    list(confusion_matrix.keys())[1]: [0, 1],
    list(confusion_matrix.keys())[2]: [0, 2],
    list(confusion_matrix.keys())[3]: [0, 3],
    list(confusion_matrix.keys())[4]: [1, 0],
    list(confusion_matrix.keys())[5]: [1, 1],
    list(confusion_matrix.keys())[6]: [1, 2],
    list(confusion_matrix.keys())[7]: [1, 3]
}
fig, axs = plt.subplots(2, 4, figsize=(15, 6))  # Create subplots with a specific size
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
    ax.set_title(now_kernel)
    if 'MI' in now_kernel or 'CI' in now_kernel:
        ax.set_xlabel('Prediction')
    if 'AF\nHigh' in now_kernel or 'MI\nHigh' in now_kernel:
        ax.set_ylabel('Labels', rotation=0)
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    'TP': np.array([(3271, 198, 557, 139), (691, 27, 131, 39)]),
    'FN': np.array([(1577, 115, 336, 117), (831, 440, 464, 546)]),
    'FP': np.array([(2338, 418, 467, 208), (1537, 111, 375, 176)]),
    'TN': np.array([(33075, 31735, 39827, 30061), (3040, 13316, 4203, 15074)])
}
fig, axs = plt.subplots(1, 4, figsize=(16, 2))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    pcm = axs[index].imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=axs[index])

    classes_x = ['AF', 'AS', 'MI', 'CI']
    classes_y = ['High', 'Low']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        axs[index].text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    axs[index].set_xticks(np.arange(len(classes_x)), classes_x)
    axs[index].set_yticks(np.arange(len(classes_y)), classes_y)
    axs[index].set_title(now_kernel)  # Update title
    axs[index].set_xlabel('Disease')
    if now_kernel == list(confusion_matrix.keys())[0]:
        axs[index].set_ylabel('Confidence Level')
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart

confusion_matrix = {
    'TP': np.array([(3271, 198, 557, 139), (691, 27, 131, 39)]),
    'FN': np.array([(1577, 115, 336, 117), (831, 440, 464, 546)]),
    'FP': np.array([(2338, 418, 467, 208), (1537, 111, 375, 176)]),
    'TN': np.array([(33075, 31735, 39827, 30061), (3040, 13316, 4203, 15074)])
}
subplots_index = {
    list(confusion_matrix.keys())[0]: [0, 0],
    list(confusion_matrix.keys())[1]: [0, 1],
    list(confusion_matrix.keys())[2]: [1, 0],
    list(confusion_matrix.keys())[3]: [1, 1]
}
fig, axs = plt.subplots(2, 2, figsize=(11, 5))  # Create subplots with a specific size
plt.rcParams.update({'font.size': 12})
for index, now_kernel in enumerate(list(confusion_matrix.keys())):
    ax = axs[subplots_index[now_kernel][0], subplots_index[now_kernel][1]]
    pcm = ax.imshow(confusion_matrix[now_kernel], interpolation='nearest', cmap='summer')  # Display matrix as pixels
    fig.colorbar(pcm, ax=ax)

    classes_x = ['AF', 'AS', 'MI', 'CI']
    classes_y = ['High', 'Low']
    for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix[now_kernel].size, 2)):
        ax.text(j, i, confusion_matrix[now_kernel][i, j], va='center', ha='center')  # Iterate matrix indices and display numbers; reshape removes a tensor dimension

    ax.set_xticks(np.arange(len(classes_x)), classes_x)
    ax.set_yticks(np.arange(len(classes_y)), classes_y)
    ax.set_title(now_kernel)  # Update title
    if now_kernel in ['TN', 'FP']:
        ax.set_xlabel('Disease')
    if now_kernel in ['TP', 'FP']:
        ax.set_ylabel('Confidence Level')
plt.tight_layout()  # Adjust layout
plt.show()  # Render chart


