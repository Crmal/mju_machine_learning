#60181891 박재민
from tensorflow import keras
# 사이킷런 ≥0.20 필수
import sklearn

assert sklearn.__version__ >= "0.20"

# 텐서플로 ≥2.0 필수
import tensorflow as tf

assert tf.__version__ >= "2.0"

# 공통 모듈 임포트
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_img():
  img = mpimg.imread('./Data/edge_detection_ex.jpg')
  img.astype(np.float)
  plt.imshow(img)
  plt.show()
  img = img.reshape((1, 720, 1280, 3))
  img = tf.constant(img, dtype=tf.float64)
  return img

def make_filter():
  weight = np.array([([-1, -2, -1],[0, 0, 0],[1, 2, 1])*3])
  weight = weight.reshape((1,3,3,3))
  weight_init = tf.constant_initializer(weight)
  return weight_init

def cnn_SAME_vaild(image, filter):
  conv2d = keras.layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=filter, padding="SAME")(image)
  plt.imshow(conv2d.numpy().reshape(720,1280), cmap='gray')
  plt.show()

def main():
  img = make_img()
  filter = make_filter()
  cnn_SAME_vaild(img, filter)

main()