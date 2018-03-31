#Part1 - Cleaning and preparing the data
#importing the required libraries
import numpy as numpy
import pandas as pd

#importing the training dataset
df = pd.read_csv(filepath_or_buffer = "/home/imcoolswap/.kaggle/competitions/titanic/train.csv", header = None, skiprows = 1);

#loading the labels
y = df.iloc[:, 1:2].values

#loading the training features
X = df.iloc[:, 2:3]
X = pd.concat([X, df.iloc[:, 4:8]], axis = 1)

#taking care of 'nan' values
X.iloc[:,2:3] = X.iloc[:,2:3].fillna(X.iloc[:,2:3].mean(), axis = 0)
X = X.values

# Encoding categorical data
# * The model needs the categorical data like [male, female] to be encoded in the form of numbers
# to be able to process it.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

#label encoding the categorical gender field
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#one hot encoding gender to avoid 1 being considered more important than 0 by ANN
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling so that we get nice circular contours and not elliptical
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN!
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with 4 neurons
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer with 3 neurons
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Adding the thrid hidden layer with 1 neuron
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN [using binary since we have two possible prediction values]
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 150)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - making predictions on actual data set
df_test = pd.read_csv(filepath_or_buffer = "/home/imcoolswap/.kaggle/competitions/titanic/test.csv", header = None, skiprows = 1);

X_test.iloc[:,2:3] = X_test.iloc[:,2:3].fillna(X_test.iloc[:,2:3].mean(), axis = 0)
X_test = X_test.values

X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])

onehotencoder_test = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder_test.fit_transform(X_test).toarray()

X_test = X_test[:, 1:]

sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)
