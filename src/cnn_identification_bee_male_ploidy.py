import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from google.colab.patches import cv2_imshow
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__

from google.colab import drive

drive.mount('content/drive')

path = '/content/drive/MyDrive/master/aplicacao/bee_ploidy.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

gerador_treinamento = ImageDataGenerator(rescale=1./255,rotation_range=7,
horizontal_flip=True, zoom_range=0.2)
dataset_treinamento = gerador_treinamento.flow_from_directory('/content/bee_ploidy/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical', shuffle = True)

gerador_teste = ImageDataGenerator(rescale=1./255)
dataset_teste = gerador_teste.flow_from_directory('/content/bee_ploidy/test_set', target_size=(64, 64), batch_size=1, class_mode='categorical', shuffle = False)

network = Sequential()
network.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))

network.add(Conv2D(32, (3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))

network.add(Flatten())

network.add(Dense(units=3137, activation='relu'))
network.add(Dense(units=3137, activation='relu'))
network.add(Dense(units=2, activation='softmax'))

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

historico = network.fit(dataset_treinamento, epochs=100)

previsoes = network.predict(dataset_teste)

previsoes = np.argmax(previsoes, axis =1)

from sklearn.metrics import accuracy_score

accuracy_score(dataset_teste.classes, previsoes)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(dataset_teste.classes, previsoes)
sns.heatmap(cm, annot=True)

from sklearn.metrics import classification_report

print(classification_report(dataset_teste.classes, previsoes))



model_json = network.to_json()
with open('network.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import save_model

network_saved = save_model(network, '/content/weights.hdf5')

with open('network.json', 'r') as json_file:
  json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')
network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


