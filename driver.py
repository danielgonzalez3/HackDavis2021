import pandas as pd
import tensorflow as tf
from tensorflow import keras 
import numpy as np

def main():
	df = pd.read_csv('dataset.csv')
	df = df.sample(frac=1).reset_index(drop=True)
	output = df.pop('Disorder')

	inputs = df.to_numpy()
	outputs = output.to_numpy()

	model = getModel()
	model.compile(optimizer='adam',
	                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	                  metrics=['accuracy'])
	callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=100)

	history = model.fit(
	    {"userInput": inputs},
	    {"output": outputs},
	        validation_split=0.2, #Use 20% of training set as validation set
	        epochs=800,
	        callbacks=[callback, ],
    )

	model.save("model")

def getModel():
    userInput = keras.Input(shape=(24, 1), name ="userInput")

    #"None" sized vectors with Len timesteps and batch size is 1 // 1D convolution layer
    conv1d_1 = keras.layers.Conv1D(48, 3, input_shape=(24, None), name="C1", activation='relu')(userInput)

    conv1d_2 = keras.layers.Conv1D(48, 3, input_shape=(24, None), name="C2", activation='relu')(userInput)
    conv1d_3 = keras.layers.Conv1D(56, 5, input_shape=(24, None), name="C3", activation='relu')(conv1d_2)

    conv1d_4 = keras.layers.Conv1D(48, 3, input_shape=(24, None), name="C4", activation='relu')(userInput)
    conv1d_5 = keras.layers.Conv1D(64, 5, input_shape=(24, None), name="C5", activation='relu')(conv1d_4)
    conv1d_6 = keras.layers.Conv1D(64, 7, input_shape=(24, None), name="C6", activation='relu')(conv1d_5)

    # Convert Matrix to single array
    f1 = keras.layers.Flatten()(conv1d_1)
    f3 = keras.layers.Flatten()(conv1d_3)
    f6 = keras.layers.Flatten()(conv1d_6)
    

    conc = keras.layers.concatenate([f1, f3, f6])

    denseMetric = keras.layers.Dense(120, name="dense_metric", activation="relu")(conc)

    # Prevents Overfitting
    drop = keras.layers.Dropout(.3)(denseMetric)

    output = keras.layers.Dense(1, name="output", activation="sigmoid")(drop)

    model = keras.Model(
        inputs=[userInput],
        outputs=[output],
    )
    return model


if __name__ == '__main__':
	main()