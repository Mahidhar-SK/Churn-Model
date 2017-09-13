# Data preprocessing
import pandas as pd

# Import dataset
dataset = pd.read_csv('Dataset.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labenc_x_1 = LabelEncoder()
x[:, 1] = labenc_x_1.fit_transform(x[:, 1]) # coz names cant be in equations, encode

labenc_x_2 = LabelEncoder()
x[:, 2] = labenc_x_2.fit_transform(x[:, 2])

ohe = OneHotEncoder(categorical_features = [1]) # coz 0,1,2 can be mistaken for priority, encode
x = ohe.fit_transform(x).toarray()


# Splitting
from sklearn.model_selection import train_test_split
# Separate trining dataset and testing dataset. 20 out of 100 dataset -> for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # suftmax for more than 2 category

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #categorical_crossentropy for more than 2 category

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)


# Predicting
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
