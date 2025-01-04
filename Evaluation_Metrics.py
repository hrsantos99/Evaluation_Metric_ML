!pip install protobuf>=3.20.3,<5
!pip install tensorflow==2.10.0

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Commented out IPython magic to ensure Python compatibility.
tf.__version__
# %load_ext tensorboard
logdir='log'

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
          y=train_labels,
          epochs=5,
          validation_data=(test_images, test_labels))

y_true=test_labels
y_pred=model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)

classes=[0,1,2,3,4,5,6,7,8,9]

con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index = classes,
                          columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

file_writer = tf.summary.create_file_writer(logdir + '/cm')

def log_confusion_matrix(epoch, logs):
  test_pred = model1.predict_classes(test_images)
  con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()
  com_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
  con_mat_df = pd.DataFrame(con_mat_norm,
                            index = classes,
                            columns = classes)
  figure = plt.figure(figsize=(8, 8))
  sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image, 0)
  with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=epoch)

  logdir='logs/images'

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  cm_callabck = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# Função para calcular métricas
def calculate_metrics(y_true, y_pred, classes):
    con_mat = confusion_matrix(y_true, y_pred, labels=classes)

    # Inicializando métricas por classe
    metrics = {}
    for i, cls in enumerate(classes):
        # VP (diagonal principal)
        VP = con_mat[i, i]
        # FN (soma da linha, excluindo VP)
        FN = np.sum(con_mat[i, :]) - VP
        # FP (soma da coluna, excluindo VP)
        FP = np.sum(con_mat[:, i]) - VP
        # VN (soma total, excluindo linha e coluna do VP)
        VN = np.sum(con_mat) - (VP + FN + FP)

        # Cálculos das métricas
        accuracy = (VP + VN) / (VP + FP + FN + VN)
        precision = VP / (VP + FP) if (VP + FP) != 0 else 0
        recall = VP / (VP + FN) if (VP + FN) != 0 else 0
        specificity = VN / (VN + FP) if (VN + FP) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # Armazenando métricas da classe
        metrics[cls] = {
            "Accuracy": round(accuracy, 2),
            "Precision": round(precision, 2),
            "Recall (Sensitivity)": round(recall, 2),
            "Specificity": round(specificity, 2),
            "F1-Score": round(f1, 2),
        }

    return metrics

# Cálculo e exibição das métricas
metrics = calculate_metrics(y_true, y_pred, classes)
for cls, values in metrics.items():
    print(f"Class {cls}:")
    for metric, value in values.items():
        print(f"  {metric}: {value}")
    print()

# Binarizar as labels verdadeiras (y_true) para ROC multi-classe
y_true_bin = label_binarize(y_true, classes=classes)
n_classes = len(classes)

# Prever probabilidades em vez de classes (necessário para ROC)
y_pred_prob = model.predict(test_images)

# Inicialização para armazenar valores
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calcular FPR, TPR e AUC para cada classe
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Gerar curva ROC para todas as classes
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})"
    )

# Linha de referência para um modelo aleatório
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Personalização do gráfico
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class Classification")
plt.legend(loc="lower right")
plt.show()
