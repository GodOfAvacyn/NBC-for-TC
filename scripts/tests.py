from sklearn.naive_bayes import BernoulliNB
import tensorflow as tf
from pygmo import hypervolume
from tensorflow import keras
from keras import backend as K
from keras import datasets, layers, models
import numpy as np

from NBC import BinaryNBC
from NBC import FeatureExctractor

X = [
    [1,0,0],
    [0,1,0],
    [1,0,0],
    [0,0,1],
    [1,0,0]
]

y = [1,0,1,0,1]

nbc = BinaryNBC(X,y)

p1 = nbc.predict([1,0,0])
p2 = nbc.predict([0,1,0])
print(p1)  # Will predict class "1"
print(p2)  # Will predict class "0"


# Example of tensorflow usage

input_shape = (1,20,20,5)

A = models.Sequential()
A.add(layers.Conv2D(5,(3,3), padding="same",activation="relu",input_shape=input_shape[1:]))
A.add(layers.Conv2D(5,(3,3), padding="same",activation="relu"))
A.add(layers.Conv2D(5,(3,3), padding="same",activation="relu"))

Q = models.Sequential()
Q.add(layers.Conv2D(5,(3,3), padding="same",activation="relu",input_shape=input_shape[1:]))
Q.add(layers.Conv2D(5,(3,3), padding="same",activation="relu"))
Q.add(layers.Conv2D(5,(3,3), padding="same",activation="relu"))

extractor = FeatureExctractor(A,Q)
ems = extractor.get_hypervolume_emissions(tf.random.normal(input_shape))
print(ems)