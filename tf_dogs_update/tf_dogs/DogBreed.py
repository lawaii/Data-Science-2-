import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
import tensorflow as tf
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.metrics import F1Score
import warnings
tf.config.run_functions_eagerly(True)

warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 10

fpath = r'C:\Users\Lawaiian\Desktop\CS\Data Science 2\deadlock\tf_dogs\tf_dogs\data\Images'
random_seed = 42

img_size = 224
batch_size = 32
train = tf.keras.utils.image_dataset_from_directory(
    fpath,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"
)

val = tf.keras.utils.image_dataset_from_directory(
    fpath,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical"
)

run = neptune.init_run(
    project="lawaiian999/DataScience2",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTMwOWM4Ny1jNGM5LTQ4NTEtOTFlNS0wN2Q3ZjA0ZGVlMTUifQ==",
)  # your credentials

class_names = train.class_names

names = []
for name in class_names:
    names.append(name.split("-")[1])

print(names[:10])  # Printing some species

plt.figure(figsize=(10, 10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")

Model_URL = 'https://kaggle.com/models/google/resnet-v2/frameworks/TensorFlow2/variations/50-classification/versions/2'
model = Sequential([
    tf.keras.layers.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)),
    hub.KerasLayer(Model_URL),
    tf.keras.layers.Dropout(0.5),  # 根据需要调整丢弃率
    tf.keras.layers.Dense(120, activation="softmax")])

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy",
             Precision(name='precision'),
             Recall(name='recall'),
             F1Score(num_classes=120, average='weighted', name='f1_score'),
             AUC(num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None,
                 multi_label=False,
                 num_labels=None,
                 label_weights=None,
                 from_logits=False)
             ]
)

model.build((img_size, img_size, 3))

model.summary()

model_name = "model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_name,
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                 verbose=1, restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.0001)


neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

history = model.fit(train, epochs=3, validation_data=val, callbacks=[checkpoint, reduce_lr])

plt.figure(figsize=(20, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['ROC'])
plt.plot(history.history['val_f1_score'])
plt.title('roc')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

loss, accuracy, precision, recall, f1_score = model.evaluate(val)

print(f"Accuracy is: {round(accuracy * 100, 2)}%")
print(f"Precision is: {round(precision * 100, 2)}%")
print(f"Recall is: {round(recall * 100, 2)}%")
print(f"F1Score is: {round(f1_score * 100, 2)}%")
model.save('resnet_model.h5')
