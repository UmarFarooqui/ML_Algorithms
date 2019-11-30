## Tree based algorithms:
ML algorithms which build decision trees from training data.

Tree based algorithms can be of two types:
1. <u>Decision tree learning algorithms.</u>  They build a single decision tree.

2. <u>Ensemble learning algorithms.</u>  They build multiple decision trees and combine their results.

### How decision tree learning works:
It works by choosing a feature which divides the training data into homogeneous subsets and applies this criterion recursively.

### Ways to measure homogeneity of the subsets
1. Information Gain
2. Gini Impurity

### Ranking important features
1. Summarize label by different features.

Say we have following features: feature1, feature2, feature3

and the label: label.

The dataframe is data.
Then,
```python
data.groupby(["feature1"])["label"].sum() * 100 / data.groupby(["feature1"])["label"].count()
```
Similarly for the other features,
```python
data.groupby(["feature2"])["label"].sum() * 100 / data.groupby(["feature2"])["label"].count()
```

2. Check which features might be more important than others. The higher the difference in the values, the more important is the feature.
3. Select relevant features.

### Building a decision tree
The following steps are needed:
1. Split the data in training and test data.
2. Build the decision tree. (Training Phase)
3. Test the decision tree. (Predictions versus actual)

Before embarking on the training phase, make sure to convert any categorical data to numerical representation .
```python
def catToNum(series):
    series = series.astype("category")    # Changes the column type to category
    return series.cat.codes    # converts to numerical value
```

And then you can apply it to all the features that have categorical data.
```python
num_data = trainingData[["feature4", "feature5"]].apply(catToNum)
```

To change the original dataframe.
```python
trainingData[["feature4", "feature5"]] = num_data
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
clf = clf.fit(train[["feature1", "feature2", "feature3", "feature4", "feature5"]], train["label"])
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
We can control the complexity of the tree by setting the hyperparameters.
e.g. max_leaf_nodes
```python
clf = DecisionTreeClassifier(max_leaf_nodes=20)
```
Calculating the feature importance after the above change might make some of the importances equal to 0, which suggest are not important.

Export the tree after that above change. It should be a simpler tree than the previous one.
Note that the bottom nodes are pruned, the top nodes are same.

### Tree Characteristics
Let's discuss the different parameters that we can control when creating a decision tree classifier.

class_weight &nbsp;&nbsp;&nbsp;Weights associated with the class labels, A value of None means equal weightage.

criterion &nbsp;&nbsp;&nbsp;Method to measure homogeneity such as gini impurity or information gain.

max_depth &nbsp;&nbsp;&nbsp;Maximum distance from root node.

max_features &nbsp;&nbsp;&nbsp;Maximum number of features to be used. The features will be picked in the order of descending importance.

max_leaf_nodes &nbsp;&nbsp;&nbsp;Maximum leaf nodes in the tree.

min_impurity_split &nbsp;&nbsp;&nbsp;Minimum percentage impurity needed in the subset to split further.

min_samples_leaf &nbsp;&nbsp;&nbsp;Minimum samples that should be present at a leaf node

min_samples_split &nbsp;&nbsp;&nbsp;Minimum samples in a subset to allow further splitting

min_weight_fraction_leaf &nbsp;&nbsp;&nbsp;Minimum fraction of samples that should be at a leaf node

Some of the parameters are related to performace of the model. These parameters are:

presort, 

random_state, 

splitter

### Measuring the accuracy of the decision tree
```python
predictions = clf.predict(test[["feature1", "feature2", "feature3", "feature4", "feature5"]])
from sklearn.metrics import accuracy_score
accuracy_score(test["label"], predictions)
```

The above accuracy score is particular to this test subset. For any other test subset, the accuracy score will change.

Another thing, we need to keep in mind while tuning the hyperparameters is that in an effort to avoid overfitting, we are not bringing the values of these hyperparameters to a point where we are having underfitting.

Overfitting and Underfitting are serious problems that need to be solved when creating any ML model.

### Overfitting and underfitting.
In machine learning, a model learns from experience i.e. the training data. In case of overfitting, it picks up patterns from the training data that are not relevant.

It is difficult to avoid overfitting since we can never fully tell which patterns are really relevant and which ones are not. The only way is to test it on as many new samples as possible.

Overfitting is also difficult to resolve because in order to avoid it, we may go too far in the other direction and introduce the problem of underfitting.

Underfitting is when we are missing patterns that are relevant to the the universe of the data.

One way to avoid overfitting is to avoid extremely complex models especially when we are working with a small dataset.

### Using Ensembles of algorithms to overcome overfitting.

In case of decision tree models, we can avoid overfitting by the following mechanisms:
1. Pruning. &nbsp;&nbsp;&nbsp;Reducing the complexity of a decision tree. 
2. Ensemble learning. &nbsp;&nbsp;&nbsp;Build multiple decision trees and combine their results.

### Pruning.
We start from the bottom of the decision tree and critically evaluate the nodes.

When we remove a node, we need to measure the accuracy of the decision tree on one or more test dataset and if the accuracy remains same or improves, we can prune the node from the tree.

So, we see that for this technique we need enough test data.

### Ensemble learning.
In ensemble learning, we build models using different
1. Techniques &nbsp;&nbsp;&nbsp;Say, gini impurity vs information gain, decision tree vs logistic regression.
2. Training Sets &nbsp;&nbsp;&nbsp;Each tree built using a different subset of training data.
3. Features &nbsp;&nbsp;&nbsp;Each tree built using a different subset of features.
4. Hyperparameters &nbsp;&nbsp;&nbsp;Each tree built using a different values of hyperparameters such as max_tree_depth e.g. 10 vs 20

Then, we combine these models using the following techniques:
1. Random Forests
2. Gradient Boosted Trees

### Random Forests.

### Gradient Boosted Trees.

Reference:
https://app.pluralsight.com/library/courses/tree-based-models-classification/table-of-contents