## Tree based algorithms:
ML algorithms which build decision trees from training data.

Tree based algorithms can be of two types:
1. Decision tree learning algorithms

They build a single decision tree.

2. Ensemble learning algorithms

They build multiple decision trees and combine their results.

### How decision tree learning works:
It works by choosing a feature which divides the training data into homogeneous subsets and applies this criterion recursively.

### Ways to measure homogeneity of the subsets
1. Information Gain
2. Gini Impurity

### Ranking important features
1. Summarize label by different features.

Say we have following features: feature1, feature2, feature3

and the label: label

The dataframe is data
Then,
```python
data.groupby(['feature1'])['label'].sum() * 100 / data.groupby(['feature1'])['label'].count()
```
Similarly for the other features,
```python
data.groupby(['feature2'])['label'].sum() * 100 / data.groupby(['feature2'])['label'].count()
```

2. Check which features might be more important than others. The higher the difference in the values, the more important is the feature.

### Building a decision tree
The following steps are needed:
1. Split the data in training and test data.
2. Build the decision tree. (Training Phase)
3. Test the decision tree. (Predictions versus actual)

Before embarking on the training phase, make sure to convert any categorical data to numerical representation .
```python
def catToNum(series):
    series = series.astype('category')    # Changes the column type to category
    return series.cat.codes    # converts to numerical value
```

And then you can apply it to all the features that have categorical data.
```python
num_data = trainingData[['feature4', 'feature5']].apply(catToNum)
```

To change the original dataframe.
```python
trainingData[['feature4', 'feature5']] = num_data
```

Also, check and drop any rows having missing values.
```python
trainingData = trainingData.dropna()
```

Split into test and train groups.
```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(trainingData, test_size=0.2)
```

Build decision tree
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(train[["feature1", "feature2", "feature3", "feature4", "feature5"]], train['label'])
```

The last line above applies the algorithm and returns another DecisionTreeClassifier object which has the decision tree embedded in it.

Visualize the decision tree
To see the importance of each feature
```python
clf.feature_importances_    # Higher values means higher importance
```

To see it visually
```python
from sklearn import tree
with open("result.dot", "w") as f:
    f = tree.export_graphviz(clf, feature_names=["feature1", "feature2", "feature3", "feature4", "feature5"], outfile=f)
```

To open the dot file created in the last step above, we need Graphviz software.
When we open it, we will see the decision nodes. This tree can be very complex to comprehend.
However, the complexity doesn't mean that the model is more accurate since it might be overfitting.

### How to control the complexity of this tree?