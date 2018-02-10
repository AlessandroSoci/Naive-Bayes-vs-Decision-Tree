# Text Classification

## Run
- download the [repository](https://github.com/AlessandroSoci/Naive-Bayes-vs-Decision-Tree/archive/master.zip) (clone or zip download)
- run `Text-Classification.py`

The script will output the results for each dataset and will create a table for better reading.

## Introduction
This project aims to compare the performances of two lerning algorithms, Naive Bayes and Decision Trees, comparing their accuracy with respect to many different datasets, showing the main characteristics of the two models. 

### Dataset
12 Datasets were used, downloaded from [UCI repository](https://archive.ics.uci.edu/ml/datasets.html) and saved in .csv format. They are:
- Iris
- Echocardiogram
- Mushroom
- Breats
- Credit
- Pima
- Hepatitis
- Wine
- Voting
- Car
- Dermatology
- Glass

## Implementation
The datasets are loaded with [**Pandas**](https://pandas.pydata.org/) package. It allows to preprocess the datasets to adapt them to the different algorithms and in order to improve the performances for some datasets. 

Among the useful *Pandas* routines:
- *get_dummies*: convert categorical variable into dummy/indicator variables
- *factorize*: encode input values as an enumerated type or categorical variable
- *drop*: return new object with labels in requested axis removed

Furthermore in some case the continuos values are normalized.

In some dataset there are unknown values denotated with '?'. 
The following strategies are applied to solve this problem:
- drop the whole line that contains '?', in case they are relatively few with respect to the whole dataset
- replace with the mean value, in case of continuos features (in this case the *Imputer* class of Scikit-learn is used)

The strategy **K-fold cross validation** is used to obtain results valid and independent from partitioing between the test set and training set.
Specifically, a 10-fold cross validation is used, ie the dataset is divided into 10 subsets, where 9 of them used as a training set and 1 as a test set. 

The accuracy if the results is an average of all the accuracies obtained with cross validation.

The learning algorithms used are implemented in the [Scikit-learn](http://scikit-learn.org/stable/) library.

## Results

|    Dataset         |   Naive Bayes    |   Decision Tree   |
|--------------------|------------------|-------------------|
| *Iris*             | 94.67 (+/- 2.91) | 94.67 (+/- 2.91)  |
| *Echocardiogram*   | 95.89 (+/- 3.15) | 94.64 (+/- 4.32)  |
| *Mushroom*         | 98.49 (+/- 2.26) | 99.22 (+/- 0.96)  |
| *Breats*           | 97.36 (+/- 0.98) | 92.97 (+/- 1.88)  |
| *Credit*           | 82.44 (+/- 6.12) | 78.62 (+/- 6.58)  |
| *Pima*             | 75.52 (+/- 2.14) | 69.52 (+/- 3.27)  |
| *Hepatitis*        | 88.46 (+/- 4.55) | 76.87 (+/- 5.28)  |
| *Wine*             | 96.11 (+/- 2.79) | 84.87 (+/- 5.54)  |
| *Voting*           | 94.71 (+/- 1.87) | 95.15 (+/- 1.43)  |
| *Car*              | 73.19 (+/- 4.90) | 88.89 (+/- 4.84)  |
| *Dermatology*      | 98.10 (+/- 1.36) | 93.20 (+/- 3.46)  |
| *Glass*            | 69.39 (+/- 14.28)| 89.68 (+/- 7.70)  |
| **Average**        | 88.70            | 88.20             |

## Conclusions
At the end of this study, with these specific datasets and dataset shuffles, altough on some datasets (Car, Glass, ...) Decision tree obtains a higher accuracy, Naive Bayes seems to perform slightly better on average.

## Requirements
| Software                                                    | Version        | Required |
| ------------------------------------------------------------|:--------------:| --------:|
| **Python**                                                  |     >= 3       |    Yes   |
| **Numpy** (Python Package)                                  |Tested on v1.13 |    Yes   |
| **Scikit-learn** (Python Package)                           |Tested on v0.19 |    Yes   |
| **Texttable** (Python Package)                              |Tested on v1.1  |    Yes   |
| **Pandas** (Python Package)                                 |Tested on v0.20 |    Yes   |
