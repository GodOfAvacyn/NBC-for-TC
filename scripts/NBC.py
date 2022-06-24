from sklearn.naive_bayes import BernoulliNB
import tensorflow as tf
from pygmo import hypervolume
from tensorflow import keras
from keras import backend as K
import numpy as np


class BinaryNBC:
    """
    A Naiive Bayesian Network class. In the 'predict' method, it will observe
    n emissions and will return the most probable single outcome.
    """

    def __init__(self, X, y):
        """
        Constructor for the NBC class.
        X:    A list of lists of the emissions, all of length n.
        y:    A list of the associated correct answers.
        """

        self.nbc = BernoulliNB()
        self.nbc.fit(X,y)

    def predict(self, x):
        """
        Takes in a single vector x and prints the predicted class.
        """

        predicted = self.nbc.predict([x])
        return predicted[0]


class FeatureExctractor:
    """
    Class that extracts features from our system.
    Key: [A,Q]
    """
    def __init__(self, FASTA, FASTQ):
        """
        Constructor for the FeatureExtractor class. Takes in the two models that it will be
        working with and outputting emissions for.
        """
        self.models = [FASTA, FASTQ]

    def get_hypervolume_emissions(self, x):
        """
        Key: [A,Q]
        Emissions: 1 if A higher than Q, 0 otherwise.
        [output A, output Q]
        """
        outs = [[],[]]   # outs[0] is FASTA, outs[1] is FASTQ
        vols = [[],[]]   # vols[0] is FASTA, vols[1] is FASTQ
        emissions = []

        for i in range(2):
            inp = self.models[i].input
            outputs = [layer.output for layer in self.models[i].layers]
            functors = [K.function([inp], [out]) for out in outputs]
            layer_outs = [func([x]) for func in functors]
            outs[i] = [np.reshape(out[0], (400,5)) for out in layer_outs]

        for i in range(2):
            for out in outs[i]:
                hv = hypervolume(out)
                ref = hv.refpoint(offset=0.1)
                vol = hv.compute(ref)
                vols[i].append(vol)
        
        for i in range(len(vols[0])):
            if (vols[0][i] > vols[1][i]):
                emissions.append(0)
            else:
                emissions.append(1)

        return emissions


"""
Algorithm:
1. Train the two models, FASTA and FASTQ.
2. Construct fe = FeatureExctractor(FASTA,FASTQ).
3. Create empty lists X and y.
4. For all (input, output) points in some second training set:
    a. SKip b&c if both FASTA and FASTQ are correct or both are incorrect.
    b. If FASTA predicts the correct value, append 1 to y. Otherwise, append 0 to y.
    c. Pass the input into our feature extractors get_hyper_volume_emissions method, append result to X.
5. Create an nbc = NBC(X,y).
6. Done!

Whenever there is a discrepancy between FASTA and FASTQ:
1. Pass the input into the feature extractor
2. Pass the features into the NBC's predict method.
3. The output will be 1 if FASTA is more likely to be correct, 0 if FASTQ is.
"""
