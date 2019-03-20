
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

seed = 7
numpy.random.seed(seed)


data=pd.read_csv("C:/Users/saitej/Desktop/data/finaldata2.csv", names = ["Time", "Activity"])
data["Time"] = data["Time"].astype(str)
data['Time'] = data['Time'].str.strip()
data['Time'] = data['Time'].str.replace("'","")
data["Activity"] = data["Activity"].astype(str)
data['Activity'] = data['Activity'].str.strip()
data['Activity'] = data['Activity'].str.replace("'","")
data["Time"] = pd.to_numeric(data["Time"])
data['Time'] =data["Time"]%86400

X=data['Time'].values
Y=data['Activity'].values

encoder=LabelEncoder()
encoder.fit(Y)
encoded_Y=encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
noo=len(dummy_y[0])

def baseline_model():
    
    model = Sequential()
    model.add(Dense(5, input_dim=1, activation='relu'))
    model.add(Dense(noo, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=2, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
