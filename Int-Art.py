import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree


def NBFunction(x, y, cv):
    model = GaussianNB()
    score_NB = cross_val_score(model, x, y.ravel(), cv=cv)
    print(score_NB)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_NB.mean() * 100, score_NB.std() * 2 * 100))


def TreeFunction(x, y, cv):
    dtree = tree.DecisionTreeClassifier()
    score_tree = cross_val_score(dtree, x, y.ravel(), cv=cv)
    print(score_tree)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score_tree.mean() * 100, score_tree.std() * 2 * 100))


i = 0
dataset = pd.read_csv('Dataset/Iris.csv')
dataset.label = pd.factorize(dataset.label)[0]

D = dataset.as_matrix()
X = D[:, :4]
Y = D[:, -1]

cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=8)


# -------------------Manual Split-------------------
# np.random.seed(9)
# np.random.shuffle(D)
# spilt_point = int(D.shape[0]*0.1)
# test = D[0:spilt_point, :]
# train = D[spilt_point:, :]
# test_X = test[:, :4]
# test_Y = (test[:, 4:]).astype(int)
# train_X = train[:, :4]
# train_Y = (train[:, 4:]).astype(int)

# -------------------Naive Baiyes --------------------

NBFunction(X, Y, cv)

# -------------------Decision Tree --------------------

TreeFunction(X, Y, cv)




