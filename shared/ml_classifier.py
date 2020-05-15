from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import argparse

np.random.seed(7)

class Utils():
    @staticmethod
    def count(x):
        from collections import Counter
        c = Counter(x)
        return c


class Dataset():
    def __init__(self, filename=None):
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.classes = ["clean","debris","background"]

        self.read(filename=filename)

    def read(self, filename):
        df = pd.read_csv(filename)
        df_train, df_test = train_test_split(df, test_size=0.2)
        np_train = df_train.values
        np_test = df_test.values
        self.trainX = np_train[:,:12]
        self.testX = np_test[:,:12]
        _trainY = np_train[:,-1]
        self.trainY = [self.classes.index(x) for x in _trainY]
        _testY = np_test[:,-1]
        self.testY = [self.classes.index(x) for x in _testY]

    def get_train_data(self):
        return [self.trainX, self.trainY]

    def get_test_data(self):
        return [self.testX, self.testY]

    def num_classes(self):
        if self.classes is None:
            return 0
        return len(self.classes)

    def num_data(self):
        return len(self.trainX)+len(self.testX)

    def info(self):
        print("No. of classes: {}".format(self.num_classes()))
        print ("Class labels: {}".format(self.classes))
        print ("Total data samples: {}".format(self.num_data()))

        if self.trainY is not None:
            print("Train samples: {}".format(len(self.trainY)))
            trainStat = Utils.count(self.trainY)
            for k in trainStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], trainStat.get(k, 0)))

        if self.testY is not None:
            print ("Test stats: {}".format(len(self.testY)))
            testStat = Utils.count(self.testY)
            for k in testStat.keys():
                print("\t {}:{} = {}".format(k, self.classes[k], testStat.get(k, 0)))

class GlacierClassifier():
    def __init__(self,estimator,output_folder):
        self.estimator_name = estimator
        self.output_folder = output_folder

    def grid_search(self, estimator, param_grid, features, targets):
        print("\nGrid search for algorithm:  {}".format(estimator))
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, verbose=10, n_jobs=6)
        grid.fit(features, targets)
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        return grid

    def train_and_evaluate(self, estimator, trainX, trainY, testX, testY):
        estimator.fit(trainX, trainY)
        print("Accuracy on train Set: ")
        print(estimator.score(trainX, trainY))
        print("Accuracy on Test Set: ")
        print(estimator.score(testX, testY))
        outputs = estimator.predict(testX)
        print("Classification Report: ")
        print(metrics.precision_recall_fscore_support(testY, outputs, average='weighted'))
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(testY, outputs))
        with open(self.output_folder+"/"+self.estimator_name+'.pkl', 'wb') as fid:
            pickle.dump(estimator, fid)

    def svm_linear(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nSVM with Linear Kernel')
        c = 0.01
        gamma = 0.0001
        if grid_search:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

        if train:
            estimator = SVC(kernel='linear', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

            # The best parameters are {'C': 0.01, 'gamma': 0.0001} with a score of 0.88

    def svm_rbf(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nSVM with RBF Kernel')
        c = 100
        gamma = 0.0001
        if grid_search:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            C_range = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
            gamma_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, "auto"]
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            c = grid.best_params_['C']
            gamma = grid.best_params_['gamma']

        if train:
            estimator = SVC(kernel='rbf', random_state=42, verbose=False, C=c, gamma=gamma)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

            # The best parameters for rbf svm are {'C': 100, 'gamma': 0.0001} with a score of 0.92

    def mlp(self, trainX, trainY, testX, testY, grid_search=False, train=True):
        print('\nMLP Neural Network')
        solver = 'adam'
        alpha = 0.000001
        learning_rate = 'adaptive'
        learning_rate_init = 0.0025
        momentum = 0.9
        hidden_layer_sizes = (256,)
        max_iter = 1000
        early_stopping = True
        if grid_search:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            solver_range = ['adam']
            alpha_range = [1e-6, 1e-5, 0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.01, 0.1, 0.5, 1, 1.5]
            learning_rate_range = [
                # 'constant', 
            'adaptive']
            max_iter_range = [200, 500, 1000]
            momentum_range = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            early_stopping_range = [
                True, 
                # False
                ]
            learning_rate_init_range = [0.0001, 0.001, 0.0025, 0.01, 0.1, 1]
            hidden_layer_sizes_range = [(100,), (100, 50), (128, 64), (256, 64), (256, 128, 64)]

            param_grid = dict(solver=solver_range, alpha=alpha_range, learning_rate=learning_rate_range,
                              learning_rate_init=learning_rate_init_range,
                              hidden_layer_sizes=hidden_layer_sizes_range, max_iter=max_iter_range,
                              momentum=momentum_range, early_stopping=early_stopping_range)
            grid = self.grid_search(estimator=estimator, param_grid=param_grid,
                                    features=trainX, targets=trainY)
            solver = grid.best_params_['solver']
            alpha = grid.best_params_['alpha']
            learning_rate = grid.best_params_['learning_rate']
            learning_rate_init = grid.best_params_['learning_rate_init']
            momentum = grid.best_params_['momentum']
            hidden_layer_sizes = grid.best_params_['hidden_layer_sizes']
            max_iter = grid.best_params_['max_iter']
            early_stopping = grid.best_params_[early_stopping]

            # The best parameters are {'alpha': 0.002, 'early_stopping': True, 
            # 'hidden_layer_sizes': (256, 128, 64), 'learning_rate': 'constant', 
            # 'learning_rate_init': 0.0025, 'max_iter': 200, 'momentum': 0.5, 
            # 'solver': 'adam'} with a score of 0.90

        if train:
            estimator = MLPClassifier(solver=solver, alpha=alpha, learning_rate=learning_rate,
                                      learning_rate_init=learning_rate_init, momentum=momentum,
                                      hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42,
                                      verbose=False, early_stopping=early_stopping)
            clf = Pipeline([
                ('clf', estimator)
            ])
            self.train_and_evaluate(clf, trainX, trainY, testX, testY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-f",
            "--input_data_file",
            type=str,
            help="Input csv file (Default data.csv)",
    )
    parser.add_argument(
            "-o",
            "--output_folder",
            type=str,
            help="Output folder location",
    )
    parser.add_argument(
            "-t",
            "--train",
            action='store_true',
            help="Set train true",
    )
    parser.add_argument(
            "-gs",
            "--grid_search",
            action='store_true',
            help="Set grid search true",
    )
    parsed_opts = parser.parse_args()
    filename = parsed_opts.input_data_file
    output_folder = parsed_opts.output_folder
    if parsed_opts.train:
        train = True
    else:
        train = False
    if parsed_opts.grid_search:
        grid_search = True
    else:
        grid_search = False
    try:
        assert(filename)
    except Exception as e:
        filename = "./data.csv"
        print(e," Using default file data.csv")
    try:
        assert(output_folder)
    except Exception as e:
        output_folder = "./saved_models"
        print(e," Using default folder ./saved_models")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    estimators = ['svm_linear', 'svm_rbf', 'mlp', 'decision_tree']  # ['decision_tree', 'svm_linear', 'svm_rbf', 'mlp']
    estimators = ['svm_rbf']
    dataset = Dataset(filename=filename)
    dataset.info()
    assert train or grid_search, "Enable the training or grid_search."
    for estimator in estimators:
        gc = GlacierClassifier(estimator,output_folder)
        if estimator == 'svm_linear':
            gc.svm_linear(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                          grid_search=grid_search, train=train)
        elif estimator == 'svm_rbf':
            gc.svm_rbf(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                       grid_search=grid_search, train=train)
        elif estimator == 'mlp':
            gc.mlp(trainX=dataset.trainX, trainY=dataset.trainY, testX=dataset.testX, testY=dataset.testY,
                   grid_search=grid_search, train=train)
        elif estimator == 'decision_tree':
            feature_names = ["B1","B2","B3","B4","B5","B6_VCID_1","B6_VCID_2","B7","B8","BQA","elevation","slope"]
            class_names = ["Clean","Debris","Background"]
            clf = tree.DecisionTreeClassifier(random_state=42, max_depth=10000)
            fig = clf.fit(dataset.trainX,dataset.trainY)
            # tree.plot_tree(fig, filled=True, rounded=True, label="none", 
            #                 feature_names=feature_names, class_names=class_names, impurity=False)
            # plt.savefig("./DecisionTree.png", dpi=150)
            with open(output_folder+'/decision_tree.pkl', 'wb') as fid:
                pickle.dump(clf, fid)
            plt.show()
        else:
            print("Unknown estimator: {}".format(estimator))
