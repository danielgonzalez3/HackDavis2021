import tensorflowjs as tfjs
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense


def main():
    df = pd.read_csv('dataset.csv')
    Y = df.pop('Disorder')
    X_data = df.to_numpy()
    Y_data = Y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.28, random_state= 0)

    # Check the dimension of the sets
    print('X_train:',np.shape(X_train))
    print('y_train:',np.shape(y_train))
    print('X_test:',np.shape(X_test))
    print('y_test:',np.shape(y_test))
    
    model = Sequential()
    model.add(Dense(60, input_dim=24, activation='relu'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(units = 16 , activation = 'sigmoid', input_shape = (24,))) 
    # Second layer: 1 neuron/perceptron that takes the input from the 1st layers and gives output as 0 or 1.Activation used is 'Hard Sigmoid'
    model.add(Dense(1, activation = 'hard_sigmoid'))

    # compiling the model
    sgd = keras.optimizers.SGD(lr=0.7, momentum=0.9, nesterov=True)

    #basic_model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

    model.fit(X_train, y_train, epochs=800)

    # Test, Loss and accuracy
    loss_and_metrics = model.evaluate(X_test, y_test)
    print('Loss = ',loss_and_metrics[0])
    print('Accuracy = ',loss_and_metrics[1])
    model.save('model')

if __name__ == '__main__':
    #main()
    model=tf.keras.models.load_model('model')
    tfjs.converters.save_keras_model(model, 'savedModel')
