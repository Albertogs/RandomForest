import pandas as pd
import numpy  as np
import pylab  as pyl

from csv import reader
from random import seed
from random import randrange
from math import sqrt
from sklearn.ensemble import RandomForestClassifier

dataset = list()
with open('sonar.all.csv', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
len(dataset)
for i in range(len(dataset[0])-1):
    for row in dataset:
        row[i]=float(row[i].strip())

class_values = set([row[-1] for row in dataset])
lookup = dict()
for i, value in enumerate(class_values):
    lookup[value] = i
for row in dataset:
    row[-1] = lookup[row[-1]]

print('dim rows =',len(dataset))
print('dim columns =', len(dataset[0]))
#print(dataset[0])

#Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
#Split a dataset based on an attribute and an attribute value
def test_split(index, value, data):
    left, right = list(), list()
    for row in data:
        if row[index]< value:
            left.append(row)
        else:
            right.append(row)
    return left, right
#Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size ==0:
                continue
            proportion = [row[-1] for row in group].count(class_value)/float(size)
            gini += (proportion * (1.0 - proportion))
    return gini   
#Select the best split point for a dataset
def get_split(data, n_features):
    #print('get_split: dim data =', len(data))
    class_values = list(set(row[-1] for row in data))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features)< n_features:
        index = randrange(len(data[0])-1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in data:
            groups = test_split(index, row[index], data)
            gini   = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
 
    return {'index':b_index, 'value': b_value, 'groups': b_groups}

#Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    #check for a no split
    if not left or not right:
        node['left']=node['right']= to_terminal(left+right)
        return
    #check for max depth
    if depth >= max_depth:
        node['left'], node['right']= to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left']= to_terminal(left)
    else:
        node['left']=get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right)<= min_size:
        node['right']= to_terminal(right)
    else:
        node['right']= get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
#build a decision tree
def build_tree(train, max_depth, min_size, n_features):
    root = get_split(dataset, n_features)#???????
    #print('1 length root = ',len(root))
    split(root, max_depth, min_size, n_features,  1)
    #print_tree(root)
    return root
def subsample(dataset, ratio):
    sample   = list()
    n_sample = round(len(dataset)*ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample
#Make prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)
# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree   = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth= max_depth, min_samples_split=min_size, max_features = n_features)
    x_data = list()
    y_data = list()
    x_test = list()
    x_data = [row[0:(len(row)-1)] for row in train]
    y_data = [row[-1] for row in train]
    x_test = [row[0:(len(row)-1)] for row in test]
    clf.fit(x_data, y_data)
    predictions2 = clf.predict(x_test)
    return(predictions, predictions2)
#Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct +=1
    return correct / float(len(actual)) * 100.0
# Split a dataset into k folds
def cross_validation_split(data, n_folds):
    #print('Entring cross_validation_split n_folds = ', n_folds)
    dat_split = list()
    print('dim  data =', len(data))
    dat_copy  = list(data)
    print('dim  dat_copy =', len(dat_copy))
    fold_size = int(len(data) / n_folds)
    #print('fold_size =', fold_size)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dat_copy))
            #print('index = ', index)
            fold.append(dat_copy.pop(index))
        dat_split.append(fold)
    #print('Exiting cross_validation_split n_folds = ')        
    return dat_split
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    folds = cross_validation_split(data, n_folds)
    #print('evaluation_algorithm: number folds = ', len(folds))
    scores  = list()
    scores2 = list()
    for f in range(len(folds)):
        #print('***********fold =',f+1,'****************')
        train_set = list(folds)
        fold      = folds[f]
        del train_set[f]
        train_set = sum(train_set, [])
        #print(' dim train_set =', len(train_set))
        test_set  = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, predicted2  = algorithm(train_set, test_set, *args)
        #print('dim predicted = ',len(predicted))
        #print('dim predicted2 = ',len(predicted2))
        actual     = [row[-1] for row in fold]
        accuracy   = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        accuracy2   = accuracy_metric(actual, predicted2)
        scores2.append(accuracy2)
    #print(' evaluation_algorithm: number folds = ', len(folds))    
    return scores, scores2

seed(1)
#load and prepare data
#evaluate algorithm
n_folds     = 5
max_depth   = 10
min_size    = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
    scores, scores2 = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print('Scores-2: %s' % scores2)
    print('Mean Accuracy-2: %.3f%%' % (sum(scores2)/float(len(scores2))))



