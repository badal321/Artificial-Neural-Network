
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

                    # Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

                    #Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#for the country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1]) #for countries
X = onehotencoder.fit_transform(X).toarray()
X=X[ : , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #Used to initialize the neural network
from keras.layers import Dense #required to build the layers
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

                            #Adding the layers
#input_dim is used only in the first hidden layer
#kernal_initializer initializes the weights randomly to values close to 0
#Dropout is udes for preventing overfitting
classifier.add(Dense(output_dim=6, activation='relu',input_dim=11,kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))#disable 10% neurons

classifier.add(Dense(output_dim=6, activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))#disable 10% neurons

classifier.add(Dense(output_dim=1, activation='sigmoid',kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))#disable 10% neurons

                              #Compile
#binary_crossentropy is used when output is non-categorical
#categorical_cross_entropy is used when output is categorical
#adam optimizer based on stochastic gradient descent is used
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fit
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Making predictions on unseen data
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
#The features should be inputted in a horizontal row and hence we use to [[]]
#Apply the same scale to this data
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction=(new_prediction>0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
                   

                    #Alternative Standard Approch
#K-fold cross validation
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

#build_classifiers():involves all the lines of code used for just building
#and not fitting 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, activation='relu',input_dim=11,kernel_initializer='uniform'))
    classifier.add(Dense(output_dim=6, activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(output_dim=1, activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#global classifier
classifier=KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

#cv=Number of divisions of training data
#njobs=-1 will involve all CPU's for fast computing
accuracies=cross_val_score(estimator= classifier, X=X_train, y=y_train, cv=10)
accuracies.mean()
accuracies.std()

                    #GRID SEARCH METHOD
#K-fold cross validation
from sklearn.model_selection import GridSearchCV 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, activation='relu',input_dim=11,kernel_initializer='uniform'))
    classifier.add(Dense(output_dim=6, activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(output_dim=1, activation='sigmoid',kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Remove the hyper parameters that need tuning
#For tuning a parameter within the build_classifier method,pass that
#parameter as argument. Example:'optimizer'
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32], 'epochs':[100,500], 'optimizer':['adam','rmsprop']}
grid_search= GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=10)

grid_search=grid_search.fit(X_train,y_train)
best_parameter= grid_search.best_params_
best_accuracy= grid_search.best_score_

#GridSearchCV: class
#grid= GridSearchCV(): object
#grid. : attribute
#GridSearchCV(parameters=-----)



