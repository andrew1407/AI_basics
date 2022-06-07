import os
import os.path as osp

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
import cv2 as cv


IMG_SIZE = (28, 28)
LETTERS_DIR = './examples'
LETTERS_PATHS = tuple(f for f in sorted(os.listdir(LETTERS_DIR)) if osp.isfile(osp.join(LETTERS_DIR, f)))

(traiend_x, trained_y), (test_x, test_y) = kr.datasets.mnist.load_data()

traiend_x = kr.utils.normalize(traiend_x, axis=1)
test_x = kr.utils.normalize(test_x, axis=1)

recognizer = kr.models.Sequential()
layers = (
  kr.layers.Flatten(input_shape=IMG_SIZE),
  kr.layers.Dense(units=128, activation=tf.nn.relu),
  kr.layers.Dense(units=128, activation=tf.nn.relu),
  kr.layers.Dense(units=10, activation=tf.nn.softmax)
)
for l in layers: recognizer.add(l)

recognizer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=('accuracy',))
recognizer.fit(traiend_x, trained_y, epochs=50)

loss, accuracy = recognizer.evaluate(test_x, test_y)
print(f'{loss=} {accuracy=}')

for img_path in LETTERS_PATHS:
  img = cv.imread(osp.join(LETTERS_DIR, img_path))[:,:,0]
  img_arr = np.invert(np.array([img]))
  prediction = recognizer.predict(img_arr)
  print(f'{img_path=}; max. prediction: {np.argmax(prediction)}')
