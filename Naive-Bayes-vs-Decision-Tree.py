import numpy as np
import pandas as pd

from texttable import Texttable

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import tree


# --------------------Functions---------------------
def GNBFunction(x, y, cvnb):
    model = GaussianNB()
    score_nb = cross_val_score(model, x, y.ravel(), cv=cvnb)
    mean = score_nb.mean() * 100
    std = score_nb.std() * 100
    accuracy = "%0.2f (+/- %0.2f)" % (mean, std)
    return accuracy, mean

def MNBFunction(x, y, cvnb):
    model = MultinomialNB()
    score_nb = cross_val_score(model, x, y.ravel(), cv=cvnb)
    mean = score_nb.mean() * 100
    std = score_nb.std() * 100
    accuracy = "%0.2f (+/- %0.2f)" % (mean, std)
    return accuracy, mean

def TreeFunction(x, y, cvt):
    dtree = tree.DecisionTreeClassifier()
    score_tree = cross_val_score(dtree, x, y.ravel(), cv=cvt)
    mean = score_tree.mean() * 100
    std = score_tree.std() * 100
    accuracy = "%0.2f (+/- %0.2f)" % (mean, std)
    return accuracy, mean


# ----------------- Initialization --------------------
t = Texttable()

scoresnb = []
scorestree = []
sdf = ["Iris", "Echocardiogram", "Mushroom", "Breats", "Credit", "Pima", "Hepatitis", "Wine", "Voting", "Car",
       "Dermatology", "Glass"]

# np.set_printoptions(threshold=np.nan)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=8)

# --------------------- Iris -------------------------
dataset_iris = pd.read_csv('Dataset/Iris.csv')
dataset_iris.label = pd.factorize(dataset_iris.label)[0]
D_iris = dataset_iris.as_matrix()
X_iris = D_iris[:, :4]
Y_iris = D_iris[:, -1]

# ----------------- Echocardiogram --------------------
dataset_echo = pd.read_csv('Dataset/Echocardiogram.csv')
erasable = dataset_echo.loc[dataset_echo['label'] == '?'].index.values
dataset_echo = dataset_echo.drop(erasable, 0)
dataset_echo = dataset_echo.drop(['f11'], 1)
D_echo = dataset_echo.as_matrix()
D_echo[D_echo == '?'] = 'NaN'
imp.fit(D_echo)
D_echo = imp.transform(D_echo)
X_echo = D_echo[:, :8].astype(np.float)
Y_echo = D_echo[:, -1].astype(np.float)

# --------------------- Mushroom ----------------------
dataset_mush = pd.read_csv('Dataset/Mushroom.csv')
erasable = dataset_mush.loc[dataset_mush['f11'] == '?'].index.values
dataset_mush = dataset_mush.drop(erasable, 0)
dataset_mush = pd.get_dummies(dataset_mush)
D_mush = dataset_mush.as_matrix()
X_mush = D_mush[:, 2:]
Y_mush = D_mush[:, 0]

# ------------- Breats Cancer Wisconsin ----------------
dataset_breast = pd.read_csv('Dataset/Breast_Cancer_Wisconsin.csv')
dataset_breast = dataset_breast.drop(['f1'], 1)
dataset_breast = dataset_breast.applymap(str)
erasable = dataset_breast.loc[dataset_breast['f7'] == '?'].index.values
dataset_breast = dataset_breast.drop(erasable, 0)
dataset_breast = pd.get_dummies(dataset_breast)
D_breats = dataset_breast.as_matrix()
X_breats = D_breats[:, :89]
Y_breats = D_breats[:, -1]


# --------------------- Credit -----------------------
dataset_credit = pd.read_csv('Dataset/Credit.csv')
tmp_data = dataset_credit.applymap(str)
for i in range(0, 15):
    erasable = tmp_data.loc[tmp_data[tmp_data.columns[i]] == '?'].index.values
    tmp_data = tmp_data.drop(erasable, 0)
    dataset_credit = dataset_credit.drop(erasable, 0)
dataset_credit.f2 = dataset_credit.f2.astype(float)
dataset_credit.f14 = dataset_credit.f14.astype(float)
dataset_credit = pd.get_dummies(dataset_credit)
D_credit = dataset_credit.as_matrix()
X_credit = D_credit[:, :-2]
Y_credit = D_credit[:, -2]

# ---------------------- Pima ------------------------
dataset_pima = pd.read_csv('Dataset/Pima.csv')
values_tmp = dataset_pima.values
min_max_scaler = preprocessing.MinMaxScaler()
norm = min_max_scaler.fit_transform(values_tmp)
dataset_pima = pd.DataFrame(norm)
dataset_pima.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'label']
dataset_pima.label = pd.factorize(dataset_pima.label)[0]
D_pima = dataset_pima.as_matrix()
X_pima = D_pima[:, :-1]
Y_pima = D_pima[:, -1]


# -------------------- Hepatitis ----------------------.
dataset_hepa = pd.read_csv('Dataset/Hepatitis.csv')
tmp_data = dataset_hepa.applymap(str)
label = pd.factorize(dataset_hepa.label)[0]
dataset_hepa = dataset_hepa.drop(['f1', 'f14', 'f15', 'f16', 'f17', 'f18'], 1)
for i in range(0, 15):
    erasable = tmp_data.loc[tmp_data[tmp_data.columns[i]] == '?'].index.values
    tmp_data = tmp_data.drop(erasable, 0)
    dataset_hepa = dataset_hepa.drop(erasable, 0)
dataset_hepa = pd.get_dummies(dataset_hepa.loc[:, dataset_hepa.columns != 'label'])
D_hepa = dataset_hepa.as_matrix()
X_hepa = D_hepa[:, 1:].astype(np.float)
Y_hepa = D_hepa[:, 0].astype(np.float)

# -------------------- Wine --------------------------
dataset_wine = pd.read_csv('Dataset/Wine.csv')
values_tmp = dataset_wine.values
min_max_scaler = preprocessing.MinMaxScaler()
norm = min_max_scaler.fit_transform(values_tmp)
dataset_wine = pd.DataFrame(norm)
dataset_wine.columns = ['label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
dataset_wine.label = pd.factorize(dataset_wine.label)[0]
D_wine = dataset_wine.as_matrix()
X_wine = D_wine[:, 1:]
Y_wine = D_wine[:, 0]

# -------------------- Voting -------------------------
dataset_voting = pd.read_csv('Dataset/Voting.csv')
dataset_voting = pd.get_dummies(dataset_voting)
D_voting = dataset_voting.as_matrix()
X_voting = D_voting[:, 2:]
Y_voting = D_voting[:, 0]


# -------------------- Car ------------------------------
dataset_car = pd.read_csv('Dataset/Car.csv')
dataset_car = dataset_car.applymap(str)
dataset_car.label = pd.factorize(dataset_car.label)[0]
dataset_car = pd.get_dummies(dataset_car)
D_car = dataset_car.as_matrix()
X_car = D_car[:, 1:]
Y_car = D_car[:, 0]

# -------------------- Dermatology ---------------------
dataset_derma = pd.read_csv('Dataset/Dermatology.csv')
dataset_derma = dataset_derma.applymap(str)
dataset_derma.label = pd.factorize(dataset_derma.label)[0]
dataset_derma = pd.get_dummies(dataset_derma)
D_derma = dataset_derma.as_matrix()
X_derma = D_derma[:, 1:]
Y_derma = D_derma[:, 0]

# --------------------- Glass --------------------------
dataset_glass = pd.read_csv('Dataset/Glass.csv')
dataset_glass.label = pd.factorize(dataset_glass.label)[0]
D_glass = dataset_glass.as_matrix()
X_glass = D_glass[:, :-1]
Y_glass = D_glass[:, -1]


# ------------------- Naive Baiyes --------------------
scoresnb.append(GNBFunction(X_iris, Y_iris, cv))
scoresnb.append(MNBFunction(X_echo, Y_echo, cv))
scoresnb.append(GNBFunction(X_mush, Y_mush, cv))
scoresnb.append(MNBFunction(X_breats, Y_breats, cv))
scoresnb.append(GNBFunction(X_credit, Y_credit, cv))
scoresnb.append(GNBFunction(X_pima, Y_pima, cv))
scoresnb.append(MNBFunction(X_hepa, Y_hepa, cv))
scoresnb.append(GNBFunction(X_wine, Y_wine, cv))
scoresnb.append(GNBFunction(X_voting, Y_voting, cv))
scoresnb.append(MNBFunction(X_car, Y_car, cv))
scoresnb.append(MNBFunction(X_derma, Y_derma, cv))
scoresnb.append(MNBFunction(X_glass, Y_glass, cv))

# ------------------- Decision Tree --------------------
scorestree.append(TreeFunction(X_iris, Y_iris, cv))
scorestree.append(TreeFunction(X_echo, Y_echo, cv))
scorestree.append(TreeFunction(X_mush, Y_mush, cv))
scorestree.append(TreeFunction(X_breats, Y_breats, cv))
scorestree.append(TreeFunction(X_credit, Y_credit, cv))
scorestree.append(TreeFunction(X_pima, Y_pima, cv))
scorestree.append(TreeFunction(X_hepa, Y_hepa, cv))
scorestree.append(TreeFunction(X_wine, Y_wine, cv))
scorestree.append(TreeFunction(X_voting, Y_voting, cv))
scorestree.append(TreeFunction(X_car, Y_car, cv))
scorestree.append(TreeFunction(X_derma, Y_derma, cv))
scorestree.append(TreeFunction(X_glass, Y_glass, cv))

# ----------------- Table Creation --------------------
t.add_rows([["Dataset", "Naive Bayes", "Decision Tree"], ["Iris", scoresnb[0][0], scorestree[0][0]]])
std_nb = scoresnb[0][1]
std_tree = scorestree[0][1]
for i in range(1, 12):
    t.add_row([sdf[i], scoresnb[i][0], scorestree[i][0]])
    std_nb = std_nb + scoresnb[i][1]
    std_tree = std_tree + scorestree[i][1]
t.add_row(["Average", std_nb/12, std_tree/12])
print(t.draw())