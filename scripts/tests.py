from sklearn.naive_bayes import BernoulliNB
import tensorflow as tf
from pygmo import hypervolume
from tensorflow import keras
from keras import backend as K
from keras import datasets, layers, models
import numpy as np

from NBC import BinaryNBC
from NBC import FeatureExctractor


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

# Should print a row of 3 emissions
print(ems)