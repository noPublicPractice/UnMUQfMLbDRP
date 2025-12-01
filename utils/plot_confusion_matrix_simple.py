import numpy as np
import matplotlib.pyplot as plt

# autumn    red-orange-yellow
# bone      grayscale (X-ray style)
# cool      cyan-magenta
# copper    black-to-copper
# flag      red-white-blue-black
# gray      grayscale
# hot       black-red-yellow-white
# hsv       HSV color space (red→yellow→green→cyan→blue→magenta→red)
# inferno   black-red-yellow
# jet       blue-cyan-yellow-red
# magma     black-red-white
# pink      black-pink-white
# plasma    green-red-yellow
# spring    magenta-yellow
# summer    green-yellow
# viridis   blue-green-yellow
# winter    blue-green
confusion_matrix = np.array([
    (3962, 2408),
    (3875, 36115)
])

fig_3 = plt.figure(figsize=(8, 4))
plt.rcParams.update({'font.size': 12})

plt.imshow(confusion_matrix, interpolation='nearest', cmap='summer')  # Display matrix as pixels
plt.colorbar()

classes_x = ['Positive', 'Negative']
classes_y = ['Positive', 'Negative']
for i, j in np.reshape([[[i, j] for j in range(len(classes_x))] for i in range(len(classes_y))], (confusion_matrix.size, 2)):  # Reshape removes one tensor dimension
    plt.text(j, i, confusion_matrix[i, j], va='center', ha='center')  # Iterate matrix indices to show values

# plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)  # Place xticks on top
plt.xticks(np.arange(len(classes_x)), classes_x)
plt.yticks(np.arange(len(classes_y)), classes_y)
plt.ylabel('Labels', rotation=0)
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()


