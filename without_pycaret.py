#!/usr/bin/env python
# coding: utf-8

# In[109]:


#!pip install mlflow --quiet
import os
import mlflow
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import scipy.special
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
#from lightgbm import LGBMRegressor
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from matplotlib import pyplot
import numpy as np
import plotly.graph_objects as go


# In[248]:

if __name__ == "__main__":


    mlflow.start_run(run_name="Without pycaret et and light gbm model")
        


# In[18]:



    df_summ=pd.read_csv("fl-cell-reduced.csv")
    df_summ = df_summ.drop("Unnamed: 0",axis=1)
    df_b=df_summ.drop(['% Silica Concentrate_std','% Silica Concentrate_var','% Silica Concentrate_max','% Silica Concentrate_min','% Silica Concentrate_skew','% Silica Concentrate_range','% Silica Concentrate_sem'],axis=1)


# In[20]:


    a=df_b['% Silica Concentrate_mean']
    b=df_b['date']


# In[21]:


    a=df_b['% Silica Concentrate_mean']
    b=df_b['date']
    sc = StandardScaler()
    df_b=df_b.set_index('date')
    sc.fit(df_b.drop(['% Silica Concentrate_mean'],axis=1))
    std = sc.transform(df_b.drop(['% Silica Concentrate_mean'],axis=1))
    pca = PCA(n_components=10)
    std=pca.fit_transform(std)
    df_c=pd.DataFrame(std)
    df_c['% Silica Concentrate_mean']=a
    df_c['date']=b
    df_c['lag-value']=df_c['% Silica Concentrate_mean'].shift(1)
    df_c = df_c.add_prefix('Component_')
#df_c


# In[22]:


    df_c=df_c.dropna()


# In[23]:



    train,test = train_test_split(df_c, test_size=0.33, random_state=42)


# In[24]:





# In[25]:


    poly = PolynomialFeatures(3)
    X_tr = poly.fit_transform(train.drop(['Component_date','Component_% Silica Concentrate_mean'],axis=1))


# In[26]:


    z = train['Component_date'].values


# In[27]:


    Xt = pd.DataFrame(X_tr,columns=poly.get_feature_names(train.drop(['Component_date','Component_% Silica Concentrate_mean'],axis=1).columns))

    
# In[28]:


    Xt = Xt.dropna().drop('1',axis=1)
#Xt


# In[29]:



    rega = AdaBoostRegressor()
    regr = RandomForestRegressor()
    reg = LinearRegression()
#estimators = [('randomforest',RandomForestRegressor()),('ada',AdaBoostRegressor()),('linear',LinearRegression()),]
#pipe = Pipeline(estimators)
    linear = reg.fit(Xt,train['Component_% Silica Concentrate_mean'])
    ada = rega.fit(Xt,train['Component_% Silica Concentrate_mean'])
    forest = regr.fit(Xt,train['Component_% Silica Concentrate_mean'])
#models = [linear,ada,forest]


# In[30]:


    model_1 = SelectFromModel(linear,threshold=0.1, prefit=True)
    model_2 = SelectFromModel(ada, prefit=True,threshold=0.1)
    model_3 = SelectFromModel(forest, prefit=True,threshold=0.1)


# In[31]:


    X_linear = model_1.transform(Xt)
    X_ada = model_2.transform(Xt)
    X_forest = model_3.transform(Xt)


# In[32]:


    X_linear = pd.DataFrame(X_linear)
    X_ada = pd.DataFrame(X_ada)
    X_forest = pd.DataFrame(X_forest)


# In[33]:


    X_new = pd.concat([X_linear,X_ada,X_forest],axis=1)


# In[34]:


    a = len(X_new.columns)
    num = []
    for i in range(a):
        num.append(str(i))


# In[35]:


    X_new.columns=num


# In[36]:


    tag=[]
    for i in X_new.columns:
        flag=1
        for j in X_new.columns:
            if(int(i)<int(j)):
                if(X_new[i][0] == X_new[j][0]):
                    tag.append(j)
                #X_new = X_new.drop(j,axis=1)
                #print(i,j,"equal",flag)
    tag=list(set(tag))
    for i in tag:
        X_new = X_new.drop(i,axis=1)
#X_new    


# In[37]:


    for i in Xt.columns:
        for j in X_new.columns:
            if(X_new[j][1] == Xt[i][1]):
                    X_new = X_new.rename(columns={j:i})
    tr = train.reset_index()                
    X_new['Component_% Silica Concentrate_mean'] = tr['Component_% Silica Concentrate_mean']


# In[38]:


#X_new


# In[39]:


    a = X_new.drop('Component_% Silica Concentrate_mean',axis=1)
    b = pd.concat([a.div(a[col], axis=0) for col in a.columns], axis=1) 


# In[40]:


    c = b.iloc[0:2744,0:9].add_suffix( '/' + X_new.columns[0]).reset_index()
    d = b.iloc[0:2744,9:18].add_suffix('/' + X_new.columns[1]).reset_index()
    e = b.iloc[0:2744,18:27].add_suffix('/' + X_new.columns[2]).reset_index()
    f = b.iloc[0:2744,27:36].add_suffix('/' + X_new.columns[3]).reset_index()
    g = b.iloc[0:2744,36:45].add_suffix('/' + X_new.columns[4]).reset_index()
    h = b.iloc[0:2744,45:54].add_suffix('/' + X_new.columns[5]).reset_index()
    i = b.iloc[0:2744,54:63].add_suffix('/' + X_new.columns[6]).reset_index()
    j = b.iloc[0:2744,63:72].add_suffix('/' + X_new.columns[7]).reset_index()
    k = b.iloc[0:2744,72:81].add_suffix('/' + X_new.columns[8]).reset_index()
    l = c.merge(d).merge(e).merge(f).merge(g).merge(h).merge(i).merge(j).merge(k)


# In[41]:


    l = l.set_index('index')
#l


# In[42]:


    l = l.drop(['Component_2/Component_2', 'Component_3/Component_3', 'Component_4/Component_4', 'Component_9/Component_9',
'Component_lag-value/Component_lag-value', 'Component_3 Component_lag-value/Component_3 Component_lag-value',
'Component_9 Component_lag-value/Component_9 Component_lag-value', 'Component_lag-value^2/Component_lag-value^2',
'Component_9 Component_lag-value^2/Component_9 Component_lag-value^2'],axis=1)


# In[43]:


    linear = reg.fit(l,train['Component_% Silica Concentrate_mean'])
    ada = rega.fit(l,train['Component_% Silica Concentrate_mean'])
    forest = regr.fit(l,train['Component_% Silica Concentrate_mean'])


# In[44]:


    model_1 = SelectFromModel(linear,threshold=0.1, prefit=True)
    model_2 = SelectFromModel(ada, prefit=True,threshold=0.1)
    model_3 = SelectFromModel(forest, prefit=True,threshold=0.1)


# In[45]:


    X_linear = model_1.transform(l)
    X_ada = model_2.transform(l)
    X_forest = model_3.transform(l)


# In[46]:


    X_linear = pd.DataFrame(X_linear)
    X_ada = pd.DataFrame(X_ada)
    X_forest = pd.DataFrame(X_forest)
    l_new = pd.concat([X_linear,X_ada,X_forest],axis=1,join='inner')
#l_new


# In[47]:


    a = len(l_new.columns)
    num = []
    for i in range(a):
        num.append(str(i))
    l_new.columns=num


# In[48]:


    tag=[]
    for i in l_new.columns:
        for j in l_new.columns:
            if(int(i)<int(j)):
            #flag=0
                for k in range(20):
                    if(l_new[i][k] == l_new[j][k]):
                        tag.append(j)
        
    tag=list(set(tag))
    for i in tag:
        l_new = l_new.drop(i,axis=1)
#l_new


# In[49]:


    for i in l.columns:
        for j in l_new.columns:
            #print(i,j)
            if(l_new[j][2] == l[i][2] and j not in l.columns):
                l_new = l_new.rename(columns={j:i})    
#l_new['Component_% Silica Concentrate_mean'] = tr['Component_% Silica Concentrate_mean']
#l_new


# In[50]:


    l_X = pd.concat([l_new,X_new],axis=1)


# In[51]:


#l_X


# In[52]:


#z


# In[53]:


    l_X['date']=z


# In[54]:


    train1,test1 = train_test_split(l_X, test_size=0.3, random_state=42)


# In[55]:



 
# get the dataset
    def get_dataset():
        return train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean']
 
 # get a list of models to evaluate
    def get_models():
        models = dict()
        models['knn'] = KNeighborsRegressor()
    #models['cart'] = DecisionTreeRegressor()
    #models['svm'] = SVR()
        models['gbr']=GradientBoostingRegressor()
        models['et']=ExtraTreesRegressor()
        #models['lightgbm']=LGBMRegressor()
        return models
 
   # evaluate a given model using cross-validation
    def evaluate_model(model, X, y):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
        return scores
 
 # define dataset
    X, y = get_dataset()
# get the models to evaluate
    models = get_models()
# evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, X, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


# In[57]:


    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    lgbm = GradientBoostingRegressor()
    model=lgbm.fit(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'])


# In[250]:


    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    r2_train_gbr = mean(cross_val_score(model,train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1),           train1['Component_% Silica Concentrate_mean'] , scoring='r2', cv=cv, n_jobs=-1, error_score='raise'))


# In[49]:


#import shap


# In[62]:


    et = ExtraTreesRegressor()
    model_1=et.fit(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'])
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    r2_train_et = mean(cross_val_score(model_1,train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), 
    train1['Component_% Silica Concentrate_mean'] , scoring='r2', cv=cv, n_jobs=-1, error_score='raise'))


# In[251]:


#explainer=shap.Explainer(model_1,train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
#shap_values=explainer(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1))


# In[252]:


#import matplotlib.pyplot as plt
#shap.plots.beeswarm(shap_values)
#plt.savefig("shap.pdf")


# In[246]:


#import seaborn as sns
#ax = sns.heatmap(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1).corr())
#plt.savefig("interaction_variables")
#ax


# For lightgbm model

# In[487]:


    predictions = model.predict(test1.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
    test1['predictions'] = predictions
    predictions = test1
#predictions['date']=test['date']
    predictions=predictions.set_index('date')
    predictions=predictions.sort_index()
#predictions


# In[489]:


#import numpy as np
    corr_matrix = np.corrcoef(predictions["Component_% Silica Concentrate_mean"], predictions["predictions"])
    corr=corr_matrix[0,1]
    r_sq=corr**2
    #r_sq


# In[64]:


    te = pd.DataFrame(test['Component_% Silica Concentrate_mean'])
    te['Component_9/Component_3'] = test['Component_9']/test['Component_3']
    te['Component_3 Component_lag-value/Component_3'] = test['Component_lag-value']
    te['Component_9 Component_lag-value^2/Component_9'] = test['Component_lag-value']**2
    te['Component_2/Component_lag-value'] = test['Component_2']/test['Component_lag-value']
    te['Component_4/Component_lag-value'] = test['Component_4']/test['Component_lag-value']
    te['Component_9/Component_lag-value'] = test['Component_9']/test['Component_lag-value']
    te['Component_9 Component_lag-value/Component_lag-value'] = test['Component_lag-value']
    te['Component_9 Component_lag-value^2/Component_lag-value'] = test['Component_lag-value']*test['Component_9']
    te['Component_3/Component_3 Component_lag-value']    =  1/test['Component_lag-value']
    te['Component_9/Component_3 Component_lag-value']  = test['Component_9']/(test['Component_lag-value']*test['Component_3'])
    te['Component_lag-value/Component_9 Component_lag-value'] = 1/test['Component_9']
    te['Component_2/Component_lag-value^2'] = test['Component_2']/(test['Component_lag-value']**2)
    te['Component_4/Component_lag-value^2'] = test['Component_4']/(test['Component_lag-value']**2)
    te['Component_9/Component_lag-value^2'] = test['Component_9']/(test['Component_lag-value']**2)
    te['Component_9/Component_9 Component_lag-value^2'] = 1/(test['Component_lag-value']**2)
    te['Component_lag-value/Component_9 Component_lag-value^2'] = 1/(test['Component_9']*test['Component_lag-value'])
    te['Component_2'] = test['Component_2']
    te['Component_3'] = test['Component_3']
    te['Component_4'] = test['Component_4']
    te['Component_9'] = test['Component_9']
    te['Component_lag-value'] = test['Component_lag-value']
    te['Component_3 Component_lag-value'] = test['Component_3']*test['Component_lag-value']
    te['Component_9 Component_lag-value'] = test['Component_9']*test['Component_lag-value']
    te['Component_lag-value^2'] = test['Component_lag-value']**2
    te['Component_9 Component_lag-value^2'] = test['Component_9']*(test['Component_lag-value']**2)
    te['date']=test['Component_date']


# In[65]:


    for col in te.columns:
        if col not in l_X.columns:
            te = te.drop(col,axis=1)        


# In[60]:


    predictions_ = model.predict(te.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
    te['predictions'] = predictions_
    predictions_ = te
#predictions['date']=test['date']
    predictions_=predictions_.set_index('date')
    predictions_=predictions_.sort_index()
#predictions_


# In[240]:



    plot_col=["Component_% Silica Concentrate_mean","predictions"]
    pd.options.plotting.backend = "plotly"
    fig = predictions_[plot_col].dropna().sort_index().plot()
    fig.write_html(os.getcwd() + "/gbr.html")


# r^2 for test data in lightgbm model

# In[61]:


#import numpy as np
    corr_matrix = np.corrcoef(predictions_["Component_% Silica Concentrate_mean"], predictions_["predictions"])
    corr=corr_matrix[0,1]
    r_sq_test_gbr=corr**2
#r_sq_test_lightgbm


# In[ ]:


    te = pd.DataFrame(test['Component_% Silica Concentrate_mean'])
    te['Component_9/Component_3'] = test['Component_9']/test['Component_3']
    te['Component_3 Component_lag-value/Component_3'] = test['Component_lag-value']
    te['Component_9 Component_lag-value^2/Component_9'] = test['Component_lag-value']**2
    te['Component_2/Component_lag-value'] = test['Component_2']/test['Component_lag-value']
    te['Component_4/Component_lag-value'] = test['Component_4']/test['Component_lag-value']
    te['Component_9/Component_lag-value'] = test['Component_9']/test['Component_lag-value']
    te['Component_9 Component_lag-value/Component_lag-value'] = test['Component_lag-value']
    te['Component_9 Component_lag-value^2/Component_lag-value'] = test['Component_lag-value']*test['Component_9']
    te['Component_3/Component_3 Component_lag-value']    =  1/test['Component_lag-value']
    te['Component_9/Component_3 Component_lag-value']  = test['Component_9']/(test['Component_lag-value']*test['Component_3'])
    te['Component_lag-value/Component_9 Component_lag-value'] = 1/test['Component_9']
    te['Component_2/Component_lag-value^2'] = test['Component_2']/(test['Component_lag-value']**2)
    te['Component_4/Component_lag-value^2'] = test['Component_4']/(test['Component_lag-value']**2)
    te['Component_9/Component_lag-value^2'] = test['Component_9']/(test['Component_lag-value']**2)
    te['Component_9/Component_9 Component_lag-value^2'] = 1/(test['Component_lag-value']**2)
    te['Component_lag-value/Component_9 Component_lag-value^2'] = 1/(test['Component_9']*test['Component_lag-value'])
    te['Component_2'] = test['Component_2']
    te['Component_3'] = test['Component_3']
    te['Component_4'] = test['Component_4']
    te['Component_9'] = test['Component_9']
    te['Component_lag-value'] = test['Component_lag-value']
    te['Component_3 Component_lag-value'] = test['Component_3']*test['Component_lag-value']
    te['Component_9 Component_lag-value'] = test['Component_9']*test['Component_lag-value']
    te['Component_lag-value^2'] = test['Component_lag-value']**2
    te['Component_9 Component_lag-value^2'] = test['Component_9']*(test['Component_lag-value']**2)
    te['date']=test['Component_date']


# In[ ]:


    for col in te.columns:
        if col not in l_X.columns:
            te = te.drop(col,axis=1) 


# r^2 for test data in et model

# In[66]:



    predictions_et = model_1.predict(te.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
    te['predictions'] = predictions_et
    predictions_et = te
#predictions['date']=test['date']
    predictions_et=predictions_et.set_index('date')
    predictions_=predictions_et.sort_index()
    corr_matrix = np.corrcoef(predictions_et["Component_% Silica Concentrate_mean"], predictions_et["predictions"])
    corr=corr_matrix[0,1]
    r_sq_test_et=corr**2
#r_sq_test_et


# In[258]:


    plot_col=["Component_% Silica Concentrate_mean","predictions"]
    pd.options.plotting.backend = "plotly"
    fig = predictions_et[plot_col].dropna().sort_index().plot()
    fig.write_html(os.getcwd() + "/et.html")


# In[259]:


    mlflow.log_metrics({"R square train gbr": r2_train_gbr, "R square test gbr": r_sq_test_gbr, "R square train et": r2_train_et, "R square test et": r_sq_test_et })


# In[260]:


#mlflow.log_artifact("shap.jpg")


# In[261]:


    mlflow.log_artifact('gbr.html')
    mlflow.log_artifact('et.html')


# In[262]:


    mlflow.sklearn.log_model(model,"gbr model without pycaret")
    mlflow.sklearn.log_model(model_1,"et model without pycaret")


# In[263]:


    mlflow.end_run()


# In[67]:





# In[68]:




# In[ ]:




