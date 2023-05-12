from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def dt_sk(train_images,train_labels):
    
    mdt = DecisionTreeClassifier(max_depth=10, min_samples_split=7)
    mdt.fit(train_images, train_labels)
    return mdt
    
def sel_dt_sk(train_images,train_labels,test_images):

    s = SelectKBest(f_classif, k=10)
    X_new = s.fit_transform(train_images, train_labels)
    X_test = s.transform(test_images)

    dt = DecisionTreeClassifier(max_depth=7, min_samples_split=4,criterion='entropy')
    dt.fit(X_new, train_labels)

    y_pred = dt.predict(X_test)
    return y_pred

def ccp(train_images, train_labels) :
    dt = DecisionTreeClassifier(ccp_alpha=0.001166666666666667) #best alpha chosen
    dt.fit(train_images, train_labels)   
    return dt
    

def rf(train_images, train_labels) :
    
    rf = RandomForestClassifier(max_depth=None,min_samples_split=10,criterion='entropy',n_estimators=100)
    rf.fit(train_images, train_labels)
    return rf

def xgboost(train_images,train_labels):

    xgb_gs =  XGBClassifier(n_jobs=-1,objective='multi:softmax', num_class=4,max_depth= 10,subsample=0.6,n_estimators=20)
    xgb_gs.fit(train_images,train_labels)
    return xgb_gs

