# Text Classification

## Introduction
In this project is shown the accuracy calculated with implementation of Naive Bayes and of Decision Tree powered by [Scikit-learn](http://scikit-learn.org/stable/), in order to compare the classification algorithms.
### Dataset
12 Dataset are used, downloaded from [UCI repository](https://archive.ics.uci.edu/ml/datasets.html) and saved in .csv format. They are:
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

## Run
- download [repository](https://github.com/AlessandroSoci/Naive-Bayes-vs-Decision-Tree/archive/master.zip)
- run *Text-Classification.py*

**IMPORTANT!** the "*Dataset*" folder must be in the same url path of Text-Classification.py.

## Implementation
The datasets are reading with the librarie of **Pandas**, that allow to modify in easy-way the whole structure. According to the dataset and to the relative information, some functions are called for changing and adapting it. These functions are very important because they let to do learning better the classificators. Some of these function are:
- *get_dummies*: convert categorical variable into dummy/indicator variables;
- *factorize*: Encode input values as an enumerated type or categorical variable;
- *drop*: Return new object with labels in requested axis removed;

Furthermore in some case the continuos values are normalized.

In some dataset there are unknown values denotated with '?'. The following strategies are applied to solve this problem:
- drop of whole line that contains '?', in case they are relatively few respect the whole dataset;
- replace with the mean value of column, in case of continuos features;

The *Imputer* class of Scikit-learn is used to replace with the mean value.

To obtain valid and independent results from partitioning in the test set and training set, the shuffle split is used as **Cross-Validation**. Specifically, a 10-fold cross validation is used, ie the dataset is divided into 10 subsets, and in my case, 8 of them used as a training set and 2 as a test set. The accuracy will then be given by an average of all possible combinations of the partitioning of the dataset.

## Results

|    Dataset         |   Naive Bayes    |   Decision Tree   |
|--------------------|------------------|-------------------|
| *Iris*           | 95.33 (+/- 3.71) | 94.33 (+/- 4.23)  |
| *Echocardiogram* | 96.00 (+/- 4.42) | 94.67 (+/- 4.99)  |
| *Mushroom*       | 99.72 (+/- 0.20) | 100.00 (+/- 0.00) |
| *Breats*         | 97.96 (+/- 1.30) | 93.21 (+/- 2.31)  |
| *Credit*         | 84.05 (+/- 4.07) | 80.76 (+/- 4.64)  |
| *Pima*           | 75.91 (+/- 1.94) | 69.61 (+/- 3.49)  |
| *Hepatitis*      | 86.43 (+/- 6.74) | 80.71 (+/- 9.20)  |
| *Wine*            | 96.67 (+/- 2.08) | 88.89 (+/- 5.96)  |
| *Voting*         | 95.63 (+/- 2.23) | 95.17 (+/- 3.03)  |
| *Car*            | 84.60 (+/- 1.78) | 96.97 (+/- 0.79)  |
| *Dermatology*     | 98.92 (+/- 0.81) | 95.00 (+/- 2.10)  |
| *Glass*           | 83.02 (+/- 3.61) | 97.91 (+/- 1.63)  |
| **Average**        | 91.185           | 90.603            |

## Conclusion
At the end of this study it is possible to conclude that the average accuracy on these specific datasets is slightly higher with Naive Bayes implementation, compared to Decision Tree. But it's not possible to declare if one is better than another.


## Requirements
| Software                                                    | Version        | Required |
| ------------------------------------------------------------|:--------------:| --------:|
| **Python**                                                  |     >= 3       |    Yes   |
| **Numpy** (Python Package)                                  |Tested on v1.13 |    Yes   |
| **Scikit-learn** (Python Package)                           |Tested on v0.19 |    Yes   |
| **Texttable** (Python Package)                              |Tested on v1.1  |    Yes   |
| **Pandas** (Python Package)                                 |Tested on v0.20 |    Yes   |