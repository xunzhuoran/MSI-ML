# Tools-python
import numpy as np
import pandas as pd
import datetime
import time
datasets = pd.read_csv(r"path") # Data to be imported.csv file （The first column is the dependent variable，the rest are independent variables）
datasets.iloc[:,1:].info()
y = pd.factorize(datasets.sp)[0]
X = datasets.iloc[:,1:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=34)

#Catboost
from catboost import CatBoostClassifier
model_cat = CatBoostClassifier(iterations=20000, depth=3, learning_rate=0.003,random_seed=34)
# cross validation
np.random.seed(34)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras
kfold = KFold(n_splits=10, shuffle=True, random_state=34)
results = cross_val_score(model_cat, X_train, y_train, cv=kfold)
print(results,'%.5f,%.5f'%(results.mean(),results.std()))

# training model
from datetime import datetime
start = datetime.now()
model_cat.fit(X_train, y_train,eval_set=(X_test,y_test),plot=True,verbose=False)
stop = datetime.now()
time_cat = stop – start
#assessment
class_cat=model_cat.predict(X_test)
prob_cat=model_cat.predict_proba(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix_cat = confusion_matrix(y_test,class_cat)
from sklearn.metrics import accuracy_score 
acc_cat=accuracy_score(y_test,class_cat)
from sklearn.metrics import roc_auc_score 
auc_cat = roc_auc_score(y_test,prob_cat,multi_class='ovr',average="weighted")
class_cat_train=model_cat.predict(X_train)
acc_cat_train=accuracy_score(y_train,class_cat_train)

#XGboost

#cross validation
from xgboost import XGBClassifier
import keras
model_XG = XGBClassifier(max_depth = 3,learning_rate = 0.08, objective = 'multi:softprob',eval_metric = 'auc',num_class = 20)
np.random.seed(34)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=34)
results = cross_val_score(model_XG, X_train, y_train, cv=kfold)

#training mode
from xgboost import XGBClassifier
model_XG = XGBClassifier(max_depth = 10,eta = 0.1,objective = 'multi:softprob',eval_metric = 'auc',num_class = 20)
from datetime import datetime
start = datetime.now()
model_XGb= model_XG.fit(X_train,y_train)
stop = datetime.now()
time_xgb = stop – start
#assessment
prob_XG = model_XGb.predict_proba(X_test)
class_XG = prob_XG.argmax(axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix_XG = confusion_matrix(y_test,class_XG)
from sklearn.metrics import accuracy_score
acc_XG = accuracy_score(y_test,class_XG)
from sklearn.metrics import roc_auc_score
auc_XG = roc_auc_score(y_test,prob_XG,multi_class='ovr')
class_XG_train=model_XG.predict(X_train)
acc_XG_train=accuracy_score(y_train,class_XG_train)

#Light GBM
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from datetime import datetime

#cross validation
model_lgb = LGBMClassifier(max_depth = 3, objective = 'multiclass',learning_rate = 0.04, num_class= 20)
np.random.seed(34)
kfold = KFold(n_splits=10, shuffle=True, random_state=34)
results = cross_val_score(model_lgb, X_train, y_train, cv=kfold)

#training model
from lightgbm import LGBMClassifier
start=datetime.now()
model_lgb = LGBMClassifier(max_depth = 3,num_leaves=100,objective = 'multiclass',learning_rate = 0.04, num_class= 20)
model_lgbm=model_lgb.fit(X_train,y_train)
stop=datetime.now()
time_lgbm = stop – start
#assessment
prob_lgbm = model_lgbm.predict_proba(X_test)
class_lgbm = prob_lgbm.argmax(axis=1)
# confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_lgbm = confusion_matrix(y_test,class_lgbm)

#CNN
# cross validation
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1) 
np.random.seed(34)
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras

def build_model():
    CNNmodel = keras.Sequential()
    keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=34)
    CNNmodel.add(layers.Conv1D(64,7,input_shape=(X_train.shape[1:]),activation='relu',padding='same')) 
    CNNmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))
    CNNmodel.add(layers.MaxPooling1D(3)) 
    # model.add(layers.Dropout(0.1)) 
    CNNmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))  
    CNNmodel.add(layers.MaxPooling1D(3))  
    CNNmodel.add(layers.Dropout(0.5)) 
    CNNmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))
    CNNmodel.add(layers.GlobalAveragePooling1D())
    # model.add(layers.Dropout(0.5)) 
    CNNmodel.add(layers.Dense(20,activation='softmax'))
    CNNmodel.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics='acc')
    #model.compile(tf.keras.optimizers.Adam(lr=2e-6,decay=1e-7） , loss='sparse_categorical_crossentropy', metrics='acc')
    return CNNmodel

model = KerasClassifier(build_fn=build_model, epochs=1000, batch_size=64, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=34)
results = cross_val_score(model, X_train, y_train, cv=kfold)
#training model
np.random.seed(34)
model = keras.Sequential()

# training model
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=34)
model.add(layers.Conv1D(64,3,input_shape=(X_train.shape[1:]),activation='relu',padding='same'))  
model.add(layers.Conv1D(64,3,activation='relu',padding='same'))
model.add(layers.MaxPooling1D(3)) 
model.add(layers.Conv1D(64,3,activation='relu',padding='same'))  
model.add(layers.MaxPooling1D(3))
model.add(layers.Dropout(0.3))  #(0.1 - 0.5)
model.add(layers.Conv1D(64,3,activation='relu',padding='same'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(20,activation='softmax'))
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics='acc')
history = model.fit(X_train,y_train , epochs= 1000 , batch_size = 64, validation_data = (X_test,y_test))
from sklearn.metrics import confusion_matrix
X_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test,X_pred.argmax(axis=1)) 

# CNN feature extraction
# cnn architecture
np.random.seed(34)
cnnmodel = keras.Sequential()
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=34)
cnnmodel.add(layers.Conv1D(64,7,input_shape=(X.shape[1:]),activation='relu',padding='same'))
cnnmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))
cnnmodel.add(layers.MaxPooling1D(3)) 
cnnmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))  
cnnmodel.add(layers.MaxPooling1D(3))
cnnmodel.add(layers.Dropout(0.3))
cnnmodel.add(layers.Conv1D(64,7,activation='relu',padding='same'))
cnnmodel.add(layers.Conv1D(4,7,activation='relu',padding='same'))
cnnmodel.add(layers.GlobalAveragePooling1D())
cnnmodel.add(layers.Dense(20,activation='softmax'))
cnnmodel.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics='acc')
CNN = cnnmodel.fit(X_train,y_train , epochs= 1000 , batch_size = 64)

# feature extraction
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,cnnmodel.predict(X_train).argmax(axis=1))
from keras import backend as K
from keras.models import load_model
layer_1_out = K.function(cnnmodel.layers[0].input, cnnmodel.layers[8].output)
a = pd.DataFrame(layer_1_out(X_train))


#DS-CNN
# cross validation
np.random.seed(34)
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras
def build_model_ds():
    np.random.seed(34)
    ds1model = keras.Sequential()
    keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=34)
ds1model.add(layers.Conv1D(64,7,input_shape=(X_train.shape[1:]),activation='relu',padding='same'))
    ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64))
    ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
    ds1model.add(layers.MaxPooling1D(3))
    ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64))
    ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
    ds1model.add(layers.MaxPooling1D(3))
    ds1model.add(layers.Dropout(0.1))
    ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64))
    ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
    ds1model.add(layers.GlobalAveragePooling1D())
    ds1model.add(layers.Dense(20,activation='softmax'))
    ds1model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics='acc')
    return ds1model
dsmodel = KerasClassifier(build_fn=build_model_ds, epochs=1000, batch_size=64, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=34)

# training model
np.random.seed(34)
ds1model = keras.Sequential()
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=34)
ds1model.add(layers.Conv1D(64,7,input_shape=(X_train.shape[1:]),activation='relu',padding='same'))
ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64))
ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
ds1model.add(layers.MaxPooling1D(3)) 
ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64)) 
ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
ds1model.add(layers.MaxPooling1D(3))
ds1model.add(layers.Dropout(0.1)) 
ds1model.add(layers.Conv1D(64,7,activation='relu',padding='same',groups=64))
ds1model.add(layers.Conv1D(64,1,activation='relu',padding='same',groups=1))
ds1model.add(layers.GlobalAveragePooling1D())
ds1model.add(layers.Dense(20,activation='softmax'))
#confusion_matrix
from sklearn.metrics import confusion_matrix
X_pred = ds1model.predict(X_test)
confusion_matrix_ds1model = confusion_matrix(y_test, X_pred.argmax(axis=1))


# Evaluation index
def evaluate(confusion_matrix):
    accu = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    column = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    line = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    recall =[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]   
    precision = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    accuracy = 0 
    Macro_P = 0  
Macro_R=0

    # accuracy
    for i in range(0,20):
        accu[i] = confusion_matrix[i][i]
        accuracy+= float(accu[i])/len(y_test)

    # Macro_R
    for i in range(0,20):
        for j in range(0,20):
            column[i]+=confusion_matrix[j][i]
        if column[i] != 0:
            recall[i]=float(accu[i])/column[i]  
    Macro_R=np.array(recall).mean()

#Macro_P
    for i in range(0,20):
        for j in range(0,20):
            line[i]+=confusion_matrix[i][j]
        if line[i] != 0:
            precision[i]=float(accu[i])/line[i]
Macro_P = np.array(precision).mean()

#Macro_F1
    Macro_F1 = (2 * (Macro_P * Macro_R)) / (Macro_P+Macro_R)
    
    #kappa
    n = np.sum(confusion_matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(confusion_matrix[0])):
        sum_po += confusion_matrix[i][i]
        row = np.sum(confusion_matrix[i, :])
        col = np.sum(confusion_matrix[:, i])
        sum_pe += row * col
    po = sum_po / n 
    pe = sum_pe / (n * n) 
    kappa = (po - pe) / (1 - pe)
    return(accuracy, Macro_R, Macro_P, Macro_F1, kappa)
evaluate(confusion_matrix_lgbm)
