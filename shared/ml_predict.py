from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pickle

np.random.seed(7)

class GlacierClassifier():
    def __init__(self):
        self.estimator = None

    def load_classifier(self, estimator):
        print("Using estimator "+str(estimator))
        if estimator == 'svm_linear':
            fname = './saved_models/svm_linear.pkl'
        elif estimator == 'svm_rbf':
            fname = './saved_models/svm_rbf.pkl'
        elif estimator == 'mlp':
            fname = './saved_models/mlp.pkl'
        elif estimator == 'decision_tree':
            fname = './saved_models/decision_tree.pkl'
        else:
            print("Unknown estimator "+str(estimator))

        with open(fname, 'rb') as fid:
            estimator = pickle.load(fid)

        self.estimator = estimator

        return estimator

    def predict(self, value):
        label = self.estimator.predict([value])
        return label

    # def predict_proba(self, value):
    #     label = self.estimator.predict_proba([value])
    #     return label

if __name__ == '__main__':
    image_filename = "img_LE07_140041_20051012_slice_71.npy"


    image_filename = "../data/slices/"+image_filename
    estimators = ['svm_linear', 'svm_rbf', 'mlp', 'decision_tree']  # ['svm_linear', 'svm_rbf', 'mlp', 'decision_tree']

    image_np = np.load(image_filename)

    gc = GlacierClassifier()

    for estimator in estimators:
        gc.load_classifier(estimator=estimator)
        label_np = np.zeros((image_np.shape[0],image_np.shape[1]))
        # alpha_np = np.zeros((image_np.shape[0],image_np.shape[1]))
        for row in range(image_np.shape[0]):
            if row % 100 == 0:
                print("Currently working on row "+str(row))
            for column in range(image_np.shape[1]):
                bands = image_np[row][column]
                label = gc.predict(bands)
                label_np[row][column] = label
                # alpha_np[row][column] = gc.predict_proba(bands)[0][label][0]
        np.save("./inference_data/"+str(estimator)+'_output.npy', label_np)
        # np.save("./inference_data/"+str(estimator)+'_alpha.npy', label_np)
        # plt.imsave(str(estimator)+'_output.jpg',label_np, cmap=plt.cm.RdYlGn)
    