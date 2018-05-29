import numpy as np
import pandas as pd
import time
import sys
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.regularizers import l2

import tensorflow as tf
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../feature/train_nn.csv')
test = pd.read_csv('../feature/test_nn.csv')

print(train.shape,test.shape)

y = train.is_pass.values
train.drop(['id', 'is_pass'], inplace=True, axis=1)
X = train.values
x, x_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def get_model(input_shape, layers, dropout = 0.2, regularization = 1e-4):
    input = Input(shape=(input_shape,))

    layer = input
    for layer_dim in layers:
        layer = Dense(layer_dim, activation = 'relu',
                      W_regularizer=l2(regularization),
                      b_regularizer=l2(regularization))(layer)
        layer = Dropout(dropout)(layer)
    
    #layer = Dropout(dropout)(layer)
    result = Dense(1, activation = 'sigmoid')(layer)


    model = Model(input=input, output=result)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[auc])

    return model

nn_layers = [128,64,32]
train_epoches = 200
batch_size = 512

from keras import backend as K  
def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return K.sum(s, axis=0)  

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = K.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = K.sum(y_pred - y_pred * y_true)  
    return FP/N  

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = K.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = K.sum(y_pred * y_true)  
    return TP/P  

model = get_model(x.shape[1], nn_layers)
print(model.summary())

callback = EarlyStopping("auc", patience=10, verbose=0, mode='auto')
model.fit(
    x, y,
    nb_epoch=train_epoches, batch_size=batch_size,
    validation_data=(x_test, y_test),
    callbacks=[callback], verbose = 1)

# Calculate total roc auc score
score = roc_auc_score(y_test, model.predict(x_test))
print("Total roc auc score = {0:0.4f}".format(score))

ids = test['id'].values
test.drop('id', inplace=True, axis=1)

def make_submission(model, ids, X_test,
                    submission_file_template = "../output/submission_nn_{}.csv"):
    submission = pd.DataFrame()
    submission["id"] = ids
    submission["is_pass"] = model.predict(X_test)
    filename = submission_file_template.format(time.strftime("%Y-%m-%d_{0:0.4f}".format(score)))
    submission.to_csv(filename, index=None)
    
    
make_submission(model,ids,test)
