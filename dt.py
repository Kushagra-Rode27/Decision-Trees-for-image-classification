import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
        self.value = value
        
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self.grow_tree(X, y)
        
    def predict(self, X):
        return [self.predict_helper(inputs) for inputs in X]

    def best_split(self, X, y,criterion='gini'):
        m = y.size
        if m < self.min_samples_split:
            return None, None
        
        if criterion == 'gini':
            best_criterion = self.gini
            best_score = np.inf
        elif criterion == 'information_gain':
            best_criterion = self.information_gain
            best_score = -np.inf
   
        best_idx, best_thr = None, None

        for idx in range(X.shape[1]):
            sorted_idx = np.argsort(X[:, idx])
            thresholds = X[sorted_idx, idx]
            classes = y[sorted_idx]         
            
            num_left = np.asarray([0] * self.n_classes)
            num_right = np.bincount(y)
            if(criterion == 'information_gain'):
                par_entropy = self.entropy(num_right)
            
            for i in range(1, m):
                
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                
                if thresholds[i] == thresholds[i - 1]:
                    continue
                
                if(criterion =='gini'):
                    score_left = best_criterion(num_left, i)
                    score_right = best_criterion(num_right, m - i)
                
                    score = (i * score_left + (m - i) * score_right) / m
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                        best_thr = (thresholds[i]) 
                else:

                    avg_entropy = best_criterion(num_left,num_right,m)
                    score = par_entropy - avg_entropy
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_thr = (thresholds[i]) 
        return best_idx, best_thr

    def gini(self, num, m):
        return 1.0 - sum((n / m) ** 2 for n in num)

    def information_gain(self,num_left,num_right, m):
       
        left_weight = np.sum(num_left) / m
        right_weight = 1 - left_weight
        left_entropy = self.entropy(num_left)
        right_entropy = self.entropy(num_right)
        child_entropy = left_weight * left_entropy + right_weight * right_entropy

        return child_entropy

    def entropy(self, num):
        total = np.sum(num)
        if total == 0:
            return 0
        p = np.asarray(np.asarray(num[num != 0]) / total)
        return -(p * np.log2(p)).sum()
         
    def grow_tree(self, X, y, depth=0):
        counts = np.bincount(y)
        predicted_class= np.argmax(counts)

        node = Node(value=predicted_class)
        
        if depth < self.max_depth:
            idx, thr = self.best_split(X, y,self.criterion)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                
                left=self.grow_tree(X_left, y_left, depth + 1)
                right=self.grow_tree(X_right, y_right, depth + 1)

                node = Node(feature=idx, threshold=thr,left=left ,right=right)
                
        return node
    
    def predict_helper(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature] < node.threshold: 
                node = node.left
            else:
                node = node.right
        return node.value

