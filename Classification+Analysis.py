
# coding: utf-8

# # Task Description
# 1. Both the training and testing data sets have the same schema.<br/>
# 2. The first column is the binary target - 0 or 1. The rest of columns are variables<br/>
# 3. Do not assume data is clean. Please include basic sanity checks on data.<br/>
# 4. Use the training data set to build the best linear and non-linear classification model.<br/>
# 5. Use the testing data set to evaluate model performance.  Use appropriate metrics.<br/>
# 6. Rank order variables in terms of importance and provide your reasoning. <br/>

# In[1]:

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from collections import Counter
get_ipython().magic('matplotlib inline')
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


# # Sanity Check

# In[2]:

# load data, check sample size
train = pd.read_csv('data_train.csv', header=None)
test = pd.read_csv('data_test.csv', header=None)
print('train data size: ', train.shape, '\ntest data size:', test.shape)


# In[3]:

# check if there are any columns with missing data
# return a list of [columns, # missing data]
print('train:', [[idx,i] for idx, i in enumerate(train.isnull().sum().tolist()) if i > 0])
print('test:', [[idx,i] for idx, i in enumerate(test.isnull().sum().tolist()) if i > 0])


# In[4]:

# separate dependent and independent variables
X_train = train.drop(0, axis=1)
y_train = train[0]
X_test = test.drop(0, axis=1)
y_test = test[0]


# ## Downsampling

# In[5]:

# check if the two classes are extramely imbalance
# class 0 : class 1 ~ 2.5 : 1, 
# pay attention to precision/recall for minority class and decide if unsampling is needed
print(Counter(train[0]))


# In[6]:

from sklearn.utils import resample
def downsampling(df, target):
    # Separate majority and minority classes
    df_majority = df[df[target]==0]
    df_minority = df[df[target]==1]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=len(df_minority),     # to match minority class
                                     random_state=1314) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled


# In[7]:

train_ds = downsampling(train, 0)
Counter(train_ds[0])


# In[8]:

# separate dependent and independent variables for downsampled data
X_train_ds = train_ds.drop(0, axis=1)
y_train_ds = train_ds[0]


# ## Check Data Type

# In[9]:

# check the data types of dependent varaibles (categorical or continous)
def check_data_type(data):
    """
    Args:
        data: pandas DataFrame
    Return:
        pandas DataFrame: column: column name | num_unique: unique number of values in the columns | values: unique values of the column
    """
    data_type = pd.DataFrame()
    for i in data.columns:
        col_res = pd.DataFrame({'columns': i, 'num_unique': len(data[i].unique()), 'values': [sorted(data[i].unique())]})
        if data_type.empty:
            data_type = col_res
        else:
            data_type = data_type.append(col_res, ignore_index=True)
    return data_type
# all the 96 variables are categorical, 
# and it looks the categorical variables have already been scaled, 
# so feature scaling is not performed here anymore
col_unique_values = check_data_type(X_train)
print('Number of Independent Variables: ', len(X_train.columns))
print(col_unique_values)


# ## Correlation Analysis 
# - there are highly correlated variables (pearson correlation > 0.8), such as <br/>
#             8-11, 13-16, 24-28, 30-33, 42-46, 48-51, 42-46 with 24-28, 47-51 with 29 -33 etc.
#     Principal Componenet Analysis (PCA) can be used to reduce dimensions while combing  
# - all the Xs' are slightly positively correlated with y, the maximum correlation is only 0.29 

# In[10]:

def plot_corr_matrix(corr):
    '''
    Plotting a diagonal correlation matrix
    Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    
    Args:
        corr: pandas DataFrame, correlation matrix
    '''
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20,20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.2, cbar_kws={"shrink": .5})


# In[11]:

corr = train.rename(columns={0:'y'}).corr()
print('maximum correlated with y', corr['y'].sort_values(ascending=False)[1])
plot_corr_matrix(corr)


# In[12]:

# take a closer look at column 30-33, 
# column 30, 33 are exactly the same, column 31, 21 are exacly the same
# so after removing duplicated columns, only 2 columns are remaining
train[[30, 31, 32, 33]].T.drop_duplicates().T.head()


# ## Remove duplicated columns
# As adjoining nearly correlated variables increases the contribution of their common underlying factor to the PCA, <br/>deplicated columns should be removed before PCA to avoid overemphasize their contribution

# In[13]:

# As applying drop_duplicates to the whole 96 needs too many recursions, 
# we can first group columns by unique values of each column
# then check if the columns with the same unique values are duplicate
# if so, remove the duplicates
col_unique_values['values'] = col_unique_values['values'].astype(str)
df_grouped_cols = pd.DataFrame(col_unique_values.groupby('values')['columns'].apply(list)).reset_index()
# store the list 
col_rm_duplicates = []
for cols in df_grouped_cols['columns'].tolist():
    if len(cols) > 1: # if there are more than 1 column have the same unique values, check duplicate
        col_rm_duplicates.extend(np.array(train[cols].T.drop_duplicates().T.columns))
    else:
        col_rm_duplicates.extend(np.array(cols))


# In[249]:

X_train_rm = X_train[col_rm_duplicates]
X_test_rm = X_test[col_rm_duplicates]
print('Number of variables after removing duplicated columns: ', X_train_rm.shape[1])


# **Correlation Matrix after Removing Duplicated Columns**
# - There are still groups of columns that are highly correlated, such as 62,68,84,75,67,61,63,83,74,90 
# - Taking a closer look of this groups, we will find the columns are only slight different in values
# - So the next step is to perform feature selection to additionally deal with the issue

# In[228]:

train_rm = X_train_rm.copy()
train_rm['y'] = y_train
corr = train_rm.corr()
print('maximum correlated with y', corr['y'].sort_values(ascending=False)[1])
plot_corr_matrix(corr)


# In[245]:

# import sys
# sys.setrecursionlimit(1500)
# df = train_rm.copy()
# high_corr_cols = [62,68,84,75,67,61,63,83,74,90]
# for i in high_corr_cols:
#     sorted_unique = dict((y,x) for x,y in dict(list(enumerate(sorted(df[i].unique())))).iteritems())
#     df[i] = df[i].replace(sorted_unique)


# # Feature Selection

# ## Colinear Variables (VIF > 5)

# In[281]:

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(df, threshhold=5.0):
    X = df.copy()
    X.columns = ['col_' + str(i) for i in X.columns]
    variables = range(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        # drop the columns with maximum VIF as a time until VIF is no longer bigger than threshhold
        maxloc = vif.index(max(vif))
        if max(vif) > threshhold:
            print('dropping ' + X[variables].columns[maxloc])
            del variables[maxloc]
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return [int(i.split('col_')[-1]) for i in X.columns[variables]]


# In[282]:

cols_rm_vif = calculate_vif_(X_train_rm)


# ## Principal Component Analysis
# - PCA is used to deal with the Curse of Dimensionality and highly correlated variables <br/>
# - Plot the cumulative variance vs. number of PCs included using train data only (first 12 PCs cover 95% of variance) <br/>
# - Use Grid Search with different models to decide the number of components

# In[214]:

# Plot Cumulative Explained Variance Ratio to check the contribution of each PC to the total variance
# from the plot, we can see that
# first 8 PCs can explain 90% of the variance, 
# first 12 PCs can explain 95% of the variance
# first 22 PCs can explain 99% of the variance
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
pca_full = PCA()
X_train_full = pca_full.fit_transform(X_train_rm)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(cumsum[:20])
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title("Cumulative Explained Variance Ratio for the First 20 PCs", fontsize=13)
plt.show()


# In[213]:

# pca = PCA(n_components=0.99)
# X_train_reduced = pca.fit_transform(X_train_rm)
# X_test_reduced = pca.transform(X_test_rm)
# train_reduced = pd.DataFrame(X_train_reduced)
# train_reduced['y'] = y_train
# pd.DataFrame(X_train_reduced).head()


# In[110]:

plt.figure(figsize=(16,4))
plt.subplot(131)
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==1], X_train_scaled[:, 1][np.array(y_train)==1], color="blue", label="1")
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==0], X_train_scaled[:, 1][np.array(y_train)==0], color="lightgreen", label="0")
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.title("PC1 vs PC2 by binary class ", fontsize=14)
plt.subplot(132)
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==1], X_train_scaled[:, 2][np.array(y_train)==1], color="blue", label="1")
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==0], X_train_scaled[:, 2][np.array(y_train)==0], color="lightgreen", label="0")
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.title("PC1 vs PC3 by binary class ", fontsize=14)
plt.subplot(133)
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==1], X_train_scaled[:, 3][np.array(y_train)==1], color="blue", label="1")
plt.scatter(X_train_scaled[:, 0][np.array(y_train)==0], X_train_scaled[:, 3][np.array(y_train)==0], color="lightgreen", label="0")
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.title("PC1 vs PC4 by binary class ", fontsize=14)


# # Classifiers

# ## Linear Model
# - Train data: comparing using (1) training data with duplicated columns removed only and (2) training data with VIF score > 5 removed plus 
# - Apply **standard scaling** to features
# - Find the best Linear Model over Linear SVM, Logistic Regression, Perceptron with different levels of regularization using **SGDClassifier** 
# - Use **Grid Search** to tune the parameters for each model, F1 score is used as the criteria (over AUC score) due to data imbalance
# - n_iter = 3000 is used to approach the full batch model (e.g. sklearn.LogisticRegression) accuracy as much as possible
# - upsampling minority class (1) using class_weight = 'balanced'
# - fi scoring (balance between precision and recall) used to find the best linear model in Grid Search

# In[299]:

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

linear_pipeline = Pipeline([('std_scaler', StandardScaler()),                             ("model", SGDClassifier(class_weight='balanced', n_iter=3000, random_state=1314))
                           ])
loss_options = ['hinge', 'log', 'squared_hinge', 'perceptron']
penalty_options = ['l2', 'elasticnet']
linear_params = {
        'model__loss': loss_options,
        'model__penalty': penalty_options
    }
grid = GridSearchCV(linear_pipeline, cv=5, n_jobs=-1, verbose=1, param_grid=linear_params, scoring='f1')


# In[263]:

# training data with duplicated columns removed only
grid.fit(X_train_rm, y_train)
print('Best Linear Model: Logistic Regression\n')
print(best_linear_model)
best_linear_model = grid.best_estimator_
predictions = best_linear_model.predict(X_train_rm)
print('----------------\nTrain data prediction:\n')
print(classification_report(y_train, predictions))
predictions = best_linear_model.predict(X_test_rm)
print('----------------\nTest data prediction:\n')
print(classification_report(y_test, predictions))


# In[300]:

## train data with VIF score > 5 removed
grid.fit(X_train_rm[cols_rm_vif], y_train)
best_linear_model_rm_vif = grid.best_estimator_
print('Best Linear Model: Logistic Regression\n')
print(best_linear_model_rm_vif)
best_linear_model = grid.best_estimator_
predictions = best_linear_model_rm_vif.predict(X_train_rm[cols_rm_vif])
print('----------------\nTrain data prediction:\n')
print(classification_report(y_train, predictions))
predictions = best_linear_model_rm_vif.predict(X_test_rm[cols_rm_vif])
print('----------------\nTest data prediction:\n')
print(classification_report(y_test, predictions))


# In[298]:

from sklearn.metrics import roc_curve, auc

def plot_roc_auc(actual, prediction):
    fpr, tpr, thresholds = roc_curve(actual, prediction[:,1])
    plt.plot(fpr, tpr,'r')
    plt.plot([0,1],[0,1],'b')
    plt.title('AUC: {}'.format(auc(fpr,tpr)))
    plt.show()  
    
plot_roc_auc(y_test, best_linear_model_rm_vif.predict_proba(X_test_rm[cols_rm_vif]))


# In[156]:

import pickle
# save best Linear Classifier - Logistic Regression
pickle.dump(best_linear_model, open('best_linear_classifier.sav', 'wb'))


# ** Linear Classifier Result Summary ** <br/>
# - The dataset has some imbalance issue (majority class: minority class ~= 2.5:1) <br/>
# - The best linear model picked up by grid search is Logistic Regression with L2 regularization: <br/> Comparing the training and testing F1 predicting scores (precision, recall and F1), we can see that the scores are pretty close, indicating that the model is neither overfitting nor underfitting
# - We can see that the minority class still have a low precision and a relatively high recall (almost the same with majority class) even after upsampling of the minority class (label 1) in training data

# ## Non-linear Model

# In[310]:

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# In[326]:

# Returns the best paramsuration for a model using crosvalidation
# and grid search
def best_params(model, name, parameters, X, y):
    print('Start search: ', name)
    grid = GridSearchCV(model, parameters, cv=5,
                       scoring="f1", verbose=1)
    grid.fit(X, y)
    best_estimator = grid.best_estimator_
    print('Best model: ', best_estimator)
 
    return [name, grid.best_score_, best_estimator]

# Returns the best model from a set of model families given
# training data using cross-validation.
def best_model(classifier_group, X, y):
    best_score = 0.0
    best_classifier = None
    classifiers = []
    for name, model, parameters in classifier_group:
        classifiers.append(best_params(model, name, parameters, X, y))
 
    for name, score, classifier in classifiers:
        print('Considering classifier... ' + name)
        if (score > best_score):
            best_score = score
            best_classifier = [name, classifier]
 
    return best_classifier[1]
 
# List of candidate family classifiers with parameters for grid
# search [name, classifier object, parameters].
def nonlinear_models():
    models = []
 
    rf_tuned_parameters = [{'criterion': ['gini', 'entropy'],
                            'max_depth' : [4, 8, 16],
                            'min_samples_leaf' : [1, 5, 10, 20]}]
    models.append(['RandomForest', RandomForestClassifier(n_jobs=-1), rf_tuned_parameters])
    
    
    gbt_tuned_parameters = [{'learning_rate': [0.1, 0.05, 0.01],
                              'max_depth': [4, 8, 16],
                              'min_samples_leaf': [1, 5, 10, 20]}]
    models.append(['GradientBoostingTree', GradientBoostingClassifier(), gbt_tuned_parameters])
    
    
    knn_tuned_parameters = [{"n_neighbors": [5, 10, 20, 50]}]
    models.append(['kNN', KNeighborsClassifier(n_jobs=-1), knn_tuned_parameters])
    
    # (no of inputs + no of outputs) ^ 0.5 + range(1,10)
    mlp_tuned_parameters = [{'learning_rate': [0.1, 0.05, 0.02, 0.01],
                             'activation': ['relu', 'logistic'],
                             "hidden_layer_sizes": [(10,1), (10,2), (15,1), (15,2),(20,1), (20,2)]}]
    models.append(['MLP', MLPClassifier(random_state=1314),  mlp_tuned_parameters])
    return models


# In[ ]:

best_model(nonlinear_models(), X_train_rm, y_train)


# In[304]:

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(15, 1), random_state=1)
model.fit(X_train, y_train)
 
# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


# In[ ]:




# ** Non-linear Classifier Summary** <br/>
# - Gradient Boosting classifier, while less prune to overfitting compared to Random Forest, 
# - Non-linear classifier (without upsampling) has a realtively high precision and a low recall compared to linear classifier (还没跑出来我猜的。。）
# - 你有没有什么别的建议的model 有木有deep learning的必要啊

# # Importance Rank

# In[ ]:

# get the feature rank using the best linear model (logistic) or non-linear model (guess MLP)

