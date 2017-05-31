import numpy as np
import pandas as pd

from collections import Counter
from copy import deepcopy
from queue import Queue
from math import floor, log
from random import randint
from scipy import stats
from sklearn.base import ClassifierMixin


class RandomForest(ClassifierMixin):

    def __init__(self, num_trees=10, d=5):
        self.num_trees = num_trees
        self.d = d

    def fit(self, X, y):
        self.trees = []
        X, y = np.array(X), np.array(y)

        shuffle_idx = np.array(range(0, y.size))
        np.random.shuffle(shuffle_idx)

        X_s, y_s = X[shuffle_idx], y[shuffle_idx]
        bag_size = floor(X_s.shape[0]/self.num_trees)

        # Bag data and train trees
        for t in range(0, self.num_trees):
            start, end = t*bag_size, min((t + 1)*bag_size, y.size)
            X_b, y_b = X_s[start: end], y_s[start: end]
            
            self.trees.append(DecisionTree(self.d))
            self.trees[t].fit(X_b, y_b)

    def predict(self, X):
        votes = [self.trees[t].predict(X) for t in range(0, self.num_trees)]
        predictions = stats.mode(np.array(votes))[0][0]

        return predictions


class DecisionTree(ClassifierMixin):
    
    def __init__(self, d=None):
        self.tree = {}
        self.nodes = 0
        self.d = d

    def fit(self, X, y):
        queue = Queue()
        queue.put([np.array(X), np.array(y), self.tree])

        while not queue.empty():
            [X_full, y, subtree] = queue.get()

            if self.d is None:
                X = X_full
            else:
                subset_idx = np.array(range(0, X_full.shape[1]))

                # Remove attributes randomly
                while subset_idx.size > self.d:
                    subset_idx = np.delete(subset_idx, randint(0, subset_idx.size - 1))

                X = X_full[:, subset_idx]

            gain_ratios = [self.gain_ratio(x, y) for x in X.T]
            max_idx = np.argmax(np.array(gain_ratios))

            # If no information, store proportion of y
            if len(X) == 0 or gain_ratios[max_idx] <= 0:
                subtree['col'] = -1
                subtree['p'], subtree['label'] = self.p(y)
                continue

            x = X[:, max_idx]
            categories = set(x)

            # If information gain is positive, column & branches of largest gain
            subtree['col'] = max_idx
            subtree['branches'] = {}
            subtree['p'], subtree['label'] = self.p(y)
            for category in categories:
                rows_c = np.where(x == category)
                x_c, y_c = x[rows_c], y[rows_c]

                subtree_c = {}
                subtree['branches'][category] = subtree_c
                
                # X_c = np.delete(X[rows_c], max_idx, axis=1)
                X_c = X_full[rows_c]

                self.nodes += 1
                queue.put([X_c, y_c, subtree_c])

    def prune(self, X, y):
        if len(self.tree) == 0:
            print("Classifier has not been trained / Classifier's tree is empty")

        should_continue = True
        iteration = 1

        while should_continue:
            tree_copy = deepcopy(self.tree)
            scores = []

            leaves = self.find_leaves()
            if len(leaves) == 0:
                should_continue = False
                continue
            
            for [parent, child] in leaves:
                col = parent['col']
                parent['col'] = -1
                scores.append(self.score(X, y))
                parent['col'] = col

            # Get current accuracy
            scores = np.array(scores)
            max_idx = np.argmax(scores)

            if scores[max_idx] >= self.score(X, y):
                [parent, child] = leaves[max_idx]

                parent['col'] = -1
                parent['branches'] = {}
            else:
                should_continue = False

            print('Iteration ', iteration, ' is complete')
            iteration += 1

        return self.score(X, y)

    def find_leaves(self):
        queue = Queue()
        queue.put([None, self.tree])
        
        leaves = []
        while not queue.empty():
            [parent, child] = queue.get()

            if child['col'] > -1:
                for bkey, branch in child['branches'].items():
                    queue.put([child, branch])
            else:
                if parent is not None:
                    leaves.append([parent, child])

        return leaves            

    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        if X.ndim == 1:
            X = np.array([X])

        for x in X:
            tree = self.tree
            
            while tree['col'] > -1:
                category = x[tree['col']]

                if category in tree['branches']:
                    tree = tree['branches'][category]
                else:
                    break

            predictions.append(tree['label'])

        return predictions

    def gain_ratio(self, x, y):
        # if x.dtype.kind in set('buifc'):

        categories = Counter(x)
        if len(categories) == 1:
            return 0

        data_len = len(x)
        gain = self.entropy(self.p(y, array=True)) # information gain
        iv = 0 #intrinsic value
        
        for category, count in categories.items():
            y_c = y[np.where(x == category)]
            p_y = self.p(y_c, array=True)
            gain -= (count/data_len)*self.entropy(p_y)
            iv -= count/data_len*log(count/data_len, 2)

        return gain/iv

    def entropy(self, p_list):
        p = np.array(p_list)
        p[p == 0] = 1 # Avoid log(0) warnings

        return -(p*np.log(p)/log(2)).sum()

    def p(self, y, array=False):
        counter = Counter(y)
        
        if array:
            return [count/len(y) for label, count in counter.items()]
        else:
            return {label: count/len(y) for label, count in counter.items()}, counter.most_common()[0][0]
