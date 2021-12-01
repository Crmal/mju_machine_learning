# from tensorflow.keras.datasets import reuters
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from sklearn.metrics import accuracy_score
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
#
Train_x = [[0.1,0.2,0.3],[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4, 0.5, 0.6]]
Train_y = [[0.2,0.3,0.4],[0.3,0.4,0.5],[0.4,0.5,0.6],[0.5, 0.6, 0.7]]
Train_x = np.array(Train_x)
Train_y = np.array(Train_y)

Train_x = np.expand_dims(Train_x,2)
Train_y = np.expand_dims(Train_y,2)


V_model = Sequential()
V_model.add(SimpleRNN(50, input_shape=(3, 1), return_sequences=True))
V_model.add(Dense(1))
V_model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
V_model.fit(Train_x,Train_y,epochs=500,  verbose=2)

# Test_X = np.array([[[0.5],[0.6],[0.7]]])
# print(V_model.predict(Test_X))

S_model = Sequential()
S_model.add(SimpleRNN(50, input_shape=(3, 1), return_sequences=True))
S_model.add(SimpleRNN(50, return_sequences=True))
S_model.add(Dense(1))
S_model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
S_model.fit(Train_x,Train_y,epochs=500,  verbose=2)

Test_X = np.array([[[0.5],[0.6],[0.7]]])
print("V_model\n",V_model.predict(Test_X))
Test_X = np.array([[[0.6],[0.7],[0.8]]])
print("S_modle\n",S_model.predict(Test_X))