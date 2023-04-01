import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


matrix = np.array([
  [82,  0,  0,  0,  2],
  [ 7, 7,  0,   2,  90],
  [ 7,  0,  0, 0, 17],
  [ 7,  1,  0,  0,  53],
  [ 19,  0,  0, 0, 108]
  ])

def plot_confusion_matrix(matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

labels = ['healthy', 'phase_1', 'phase_2', 'phase_3', 'phase_4']
plot_confusion_matrix(matrix, labels)