#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mlflow
import mlflow.sklearn


# In[3]:


mlflow.start_run()
mlflow.end_run()


# In[4]:


#opening zip file which contains contains the csv file
import shutil
import os
fil = os.getcwd()
shutil.unpack_archive('fl_cell_big.zip',fil)


# In[4]:


mlflow.start_run(run_name="Without pycaret et and light gbm model")


# In[2]:


import pandas as pd
df = pd.read_csv('fl_cell_big.csv')
print(df)


# In[3]:


df_mean = df.groupby('date').mean()
df_mean = df_mean.add_suffix('_mean')
df_mean = df_mean.reset_index()

df_std = df.groupby('date').std()
df_std = df_std.add_suffix('_std')
df_std = df_std.reset_index()

df_median = df.groupby('date').median()
df_median = df_median.add_suffix('_median')
df_median = df_median.reset_index()

df_var = df.groupby('date').var()
df_var = df_var.add_suffix('_var')
df_var = df_var.reset_index()

df_min = df.groupby('date').min()
#df_min = df_min.add_suffix('_min')
#df_min = df_min.reset_index()

df_max = df.groupby('date').max()
#df_max = df_max.add_suffix('_max')
#df_max = df_max.reset_index()

df_skew = df.groupby('date').skew()
df_skew = df_skew.add_suffix('_skew')
df_skew = df_skew.reset_index()

df_sem = df.groupby('date').sem()
df_sem = df_sem.add_suffix('_sem')
df_sem = df_sem.reset_index()


df_range = df_max.sub(df_min)
df_min = df_min.add_suffix('_min')
df_max = df_max.add_suffix('_max')
df_max = df_max.reset_index()
df_min = df_min.reset_index()
df_range = df_range.add_suffix('_range')
df_range = df_range.reset_index()


# In[4]:


df_summ=df_mean.merge(df_std,on='date').merge(df_median,on='date').merge(df_var,on='date').merge(df_min,on='date').merge(df_max,on='date').merge(df_skew,on='date').merge(df_sem,on='date').merge(df_range,on='date')


# In[7]:


df_b=df_summ.drop(['% Silica Concentrate_std','% Silica Concentrate_var','% Silica Concentrate_max','% Silica Concentrate_min','% Silica Concentrate_skew','% Silica Concentrate_range','% Silica Concentrate_sem'],axis=1)


# In[8]:


a=df_b['% Silica Concentrate_mean']
b=df_b['date']


# In[9]:


a=df_b['% Silica Concentrate_mean']
b=df_b['date']
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
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
df_c


# In[10]:


df_c=df_c.dropna()


# In[11]:


from sklearn.model_selection import train_test_split
train,test = train_test_split(df_c, test_size=0.33, random_state=42)


# In[12]:


from sklearn.preprocessing import PolynomialFeatures
import scipy.special
from pandas import DataFrame


# In[13]:


poly = PolynomialFeatures(3)
X_tr = poly.fit_transform(train.drop(['Component_date','Component_% Silica Concentrate_mean'],axis=1))


# In[14]:


z = train['Component_date'].values


# In[15]:


Xt = pd.DataFrame(X_tr,columns=poly.get_feature_names(train.drop(['Component_date','Component_% Silica Concentrate_mean'],axis=1).columns))


# In[16]:


Xt = Xt.dropna().drop('1',axis=1)
Xt


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
rega = AdaBoostRegressor()
regr = RandomForestRegressor()
reg = LinearRegression()
#estimators = [('randomforest',RandomForestRegressor()),('ada',AdaBoostRegressor()),('linear',LinearRegression()),]
#pipe = Pipeline(estimators)
linear = reg.fit(Xt,train['Component_% Silica Concentrate_mean'])
ada = rega.fit(Xt,train['Component_% Silica Concentrate_mean'])
forest = regr.fit(Xt,train['Component_% Silica Concentrate_mean'])
#models = [linear,ada,forest]


# In[18]:


model_1 = SelectFromModel(linear,threshold=0.1, prefit=True)
model_2 = SelectFromModel(ada, prefit=True,threshold=0.1)
model_3 = SelectFromModel(forest, prefit=True,threshold=0.1)


# In[19]:


X_linear = model_1.transform(Xt)
X_ada = model_2.transform(Xt)
X_forest = model_3.transform(Xt)


# In[20]:


X_linear = pd.DataFrame(X_linear)
X_ada = pd.DataFrame(X_ada)
X_forest = pd.DataFrame(X_forest)


# In[21]:


X_new = pd.concat([X_linear,X_ada,X_forest],axis=1)


# In[22]:


a = len(X_new.columns)
num = []
for i in range(a):
    num.append(str(i))


# In[23]:


X_new.columns=num


# In[24]:


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
X_new    


# In[25]:


for i in Xt.columns:
    for j in X_new.columns:
        if(X_new[j][1] == Xt[i][1]):
                X_new = X_new.rename(columns={j:i})
tr = train.reset_index()                
X_new['Component_% Silica Concentrate_mean'] = tr['Component_% Silica Concentrate_mean']


# In[26]:


X_new


# In[27]:


a = X_new.drop('Component_% Silica Concentrate_mean',axis=1)
b = pd.concat([a.div(a[col], axis=0) for col in a.columns], axis=1) 


# In[28]:


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


# In[29]:


l = l.set_index('index')
l


# In[30]:


l = l.drop(['Component_2/Component_2', 'Component_3/Component_3', 'Component_4/Component_4', 'Component_9/Component_9',
       'Component_lag-value/Component_lag-value', 'Component_3 Component_lag-value/Component_3 Component_lag-value',
       'Component_9 Component_lag-value/Component_9 Component_lag-value', 'Component_lag-value^2/Component_lag-value^2',
       'Component_9 Component_lag-value^2/Component_9 Component_lag-value^2'],axis=1)


# In[31]:


linear = reg.fit(l,train['Component_% Silica Concentrate_mean'])
ada = rega.fit(l,train['Component_% Silica Concentrate_mean'])
forest = regr.fit(l,train['Component_% Silica Concentrate_mean'])


# In[32]:


model_1 = SelectFromModel(linear,threshold=0.1, prefit=True)
model_2 = SelectFromModel(ada, prefit=True,threshold=0.1)
model_3 = SelectFromModel(forest, prefit=True,threshold=0.1)


# In[33]:


X_linear = model_1.transform(l)
X_ada = model_2.transform(l)
X_forest = model_3.transform(l)


# In[34]:


X_linear = pd.DataFrame(X_linear)
X_ada = pd.DataFrame(X_ada)
X_forest = pd.DataFrame(X_forest)
l_new = pd.concat([X_linear,X_ada,X_forest],axis=1,join='inner')
l_new


# In[35]:


a = len(l_new.columns)
num = []
for i in range(a):
    num.append(str(i))
l_new.columns=num


# In[36]:


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
l_new


# In[37]:


for i in l.columns:
    for j in l_new.columns:
        #print(i,j)
        if(l_new[j][2] == l[i][2] and j not in l.columns):
                l_new = l_new.rename(columns={j:i})    
#l_new['Component_% Silica Concentrate_mean'] = tr['Component_% Silica Concentrate_mean']
l_new


# In[38]:


l_X = pd.concat([l_new,X_new],axis=1)


# In[39]:


l_X


# In[40]:


z


# In[41]:


l_X['date']=z


# In[42]:


train1,test1 = train_test_split(l_X, test_size=0.3, random_state=42)


# In[43]:


from lightgbm import LGBMRegressor


# In[44]:


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
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))


# In[45]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gbr = GradientBoostingRegressor()
model=gbr.fit(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'])


# In[47]:


cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
r2_train_gbr = mean(cross_val_score(model,train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'] , scoring='r2', cv=cv, n_jobs=-1, error_score='raise'))


# In[48]:





# In[49]:


et = ExtraTreesRegressor()
model_1=et.fit(train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'])
cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
r2_train_et = mean(cross_val_score(model_1,train1.drop(['Component_% Silica Concentrate_mean','date'],axis=1), train1['Component_% Silica Concentrate_mean'] , scoring='r2', cv=cv, n_jobs=-1, error_score='raise'))


# In[51]:




# In[52]:




# In[53]:





# In[54]:





# For gbr model

# In[55]:


prediction = model.predict(test1.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
predictions = test1
predictions['predictions'] = prediction
predictions = test1
#predictions['date']=test['date']
predictions=predictions.set_index('date')
predictions=predictions.sort_index()
predictions


# In[56]:


import numpy as np
corr_matrix = np.corrcoef(predictions["Component_% Silica Concentrate_mean"], predictions["predictions"])
corr=corr_matrix[0,1]
r_sq=corr**2
r_sq


# In[57]:


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


# In[58]:


for col in te.columns:
    if col not in l_X.columns:
        te = te.drop(col,axis=1)        


# In[59]:


predictions_ = model.predict(te.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
te['predictions'] = predictions_
predictions_ = te
#predictions['date']=test['date']
predictions_=predictions_.set_index('date')
predictions_=predictions_.sort_index()
predictions_


# In[60]:


import plotly.graph_objects as go
plot_col=["Component_% Silica Concentrate_mean","predictions"]
pd.options.plotting.backend = "plotly"
fig = predictions_[plot_col].dropna().sort_index().plot()
fig.write_html("gbr.html")


# r^2 for test data in gbr model

# In[61]:


import numpy as np
corr_matrix = np.corrcoef(predictions_["Component_% Silica Concentrate_mean"], predictions_["predictions"])
corr=corr_matrix[0,1]
r_sq_test_gbr=corr**2
r_sq_test_gbr


# In[62]:


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


# In[63]:


for col in te.columns:
    if col not in l_X.columns:
        te = te.drop(col,axis=1) 


# r^2 for test data in et model

# In[64]:


import numpy as np
predictions_et = model_1.predict(te.drop(['Component_% Silica Concentrate_mean','date'],axis=1))
te['predictions'] = predictions_et
predictions_et = te
#predictions['date']=test['date']
predictions_et=predictions_et.set_index('date')
predictions_=predictions_et.sort_index()
corr_matrix = np.corrcoef(predictions_et["Component_% Silica Concentrate_mean"], predictions_et["predictions"])
corr=corr_matrix[0,1]
r_sq_test_et=corr**2
r_sq_test_et


# In[65]:


plot_col=["Component_% Silica Concentrate_mean","predictions"]
pd.options.plotting.backend = "plotly"
fig = predictions_et[plot_col].dropna().sort_index().plot()
fig.write_html("et.html")


# In[259]:


mlflow.log_metrics({"R square train gbr": r2_train_gbr, "R square test gbr": r_sq_test_gbr, "R square train et": r2_train_et, "R square test et": r_sq_test_et })


# In[261]:


mlflow.log_artifact('et.html')
mlflow.log_artifact('gbr.html')


# In[262]:


mlflow.sklearn.log_model(model,"gbr model without pycaret")
mlflow.sklearn.log_model(model_1,"et model without pycaret")


# In[5]:


mlflow.end_run()


# In[67]:





# In[68]:





# In[71]:





# In[ ]:




