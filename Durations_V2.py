#!/usr/bin/env python
# coding: utf-8

# # **DURATIONS PREDICTION FOR SLZ AT VIC (SPAIN)**

# # **Research Questions**
# 
# 1. What are the significant factors that policy makers should consider when defining parking durations?
# 
# 2. How do operational and location factors explain durations prediction?
# 
# 3. How can LZ be effectly managed based on parking durations understanding and forecast?
# 
# 4. From a modelling perspective, how should supply and demand perspectives complement to a better use of the curbside?

# # **Libraries Installation**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wilcoxon
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('python --version')
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as sm
import os
import warnings
warnings.filterwarnings('ignore')
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from patsy import dmatrices
import statsmodels.api as sm
get_ipython().system('pip install fitter')
from fitter import Fitter, get_common_distributions, get_distributions

print('Pandas', pd.__version__)
print('NumPy', np.__version__)
print('Matplotlib', matplotlib.__version__)
from pprint import pprint

get_ipython().system('pip install hillmaker')

import hillmaker as hm

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


# In[2]:


conda list


# In[3]:


operations_2018 = pd.read_excel('operations_2018.xlsx')
operations_2019 = pd.read_excel('operations_2019.xlsx')
operations = pd.concat([operations_2018,operations_2019],ignore_index=True)


# In[4]:


occ_gen = pd.read_csv('Occ_dur_ZL_pf_occupancy_dow_binofday.csv')
occ_LZ_PA = pd.read_csv('Occ_dur_ZL_pf_occupancy_ZoneId_ProfessionalActivity_dow_binofday.csv')
occ_VT = pd.read_csv('Occ_dur_vt_occupancy_VehicleType_dow_binofday.csv')
occ_dur = pd.read_csv('Occ_dur_full_arrivals_Duration_dow_binofday.csv')
vic_weather = pd.read_csv('vic_weather.csv')


# In[5]:


occ_LZ_PA_full = pd.read_csv('Occ_dur_ZL_pf_bydatetime_ZoneId_ProfessionalActivity_datetime.csv')
occ_VT_full = pd.read_csv('Occ_dur_vt_bydatetime_VehicleType_datetime.csv')
occ_dur_full = pd.read_csv('Occ_dur_full_bydatetime_Duration_datetime.csv')
occupation = pd.read_csv('Occ_dur_ZL_pf_bydatetime_datetime.csv')


# # **Databases descriptive analysis**

# In[6]:


operations.shape


# In[7]:


operations.info()


# In[8]:


operations.describe()


# In[9]:


slz_names = {'3BEAFB2A-4678-4785-A397-90CC4C95829E':'LZ1','7908475C-E4D2-45F4-9CE9-DB6F28129C48':'LZ2', 
               '051AA64E-9530-471F-AA58-B52D3BABE969':'LZ3', '7FEBCAEB-976F-4B1B-A31B-6F15EDE76D17':'LZ4',
               '326FB7A3-3A9F-472D-9FF9-348B0DD06206':'LZ5', '0F698316-47D1-470B-803A-478F6A4F529A':'LZ6',
               'CC2E43E1-F9C4-4269-AA13-AC20AE02140E':'LZ7', '598218CC-0AF2-4FD2-92C3-9160D83E5AD1':'LZ8'}
professional_activity = { 0:'Unspecified',1:'Install & Maintenance',2:'Transport and parcels',
                          3:'Construction',4:'Local Commerce',5: 'Commercial Agent',6:'Food',7:'Automotive',8:'Others' }
emissions = {0: 'High Emissions',1: 'Medium Emissions',2: 'Low Emissions',3: 'Hybrid',4: 'eVehicle'}
vehicle_type = {0:'Private',1: 'LV',2:'HV',3:'Truck',4: 'Van'}

#Categorical variables labeling
operations['ZoneId'] = operations['ZoneId'].map(slz_names)
operations['ProfessionalActivity'] = operations['ProfessionalActivity'].map(professional_activity)
operations['Emission'] = operations['Emission'].map(emissions)
operations['VehicleType'] = operations['VehicleType'].map(vehicle_type)

#Dates to datetime format
operations['TimeFinish'] = pd.to_datetime(operations.loc[:,'TimeFinish'])
operations['TimeStart'] = pd.to_datetime(operations.loc[:,'TimeStart'])
operations['TimeLimit'] = pd.to_datetime(operations.loc[:,'TimeLimit'])

operations['Year'] = operations['TimeStart'].dt.year

#Duration variable inclusion
operations['Duration'] = operations['TimeFinish'] - operations['TimeStart']
operations['Duration'] = operations.loc[:,'Duration'].dt.total_seconds()/60


# In[10]:


sns.countplot(data=operations,x='ZoneId',order=['LZ1','LZ2','LZ3','LZ4','LZ5','LZ6','LZ7','LZ8'],hue='Year')


# In[11]:


sns.countplot(data=operations,x='ZoneId',hue='ProfessionalActivity',order=['LZ1','LZ2','LZ3','LZ4','LZ5','LZ6','LZ7','LZ8'])
plt.legend()
sns.set(rc={'figure.figsize':(20,20)})


# In[12]:


outlyers = operations[operations['Duration']>150]
outlyers


# In[13]:


durations = operations[(operations['Duration']<45)&(operations['ClosedBy']=='User')]


# In[14]:


sns.distplot(durations['Duration'])
sns.set(style="whitegrid", font_scale=6)


# In[15]:


operations['Duration'].count()


# In[16]:


operations['Duration'].describe()


# In[17]:


durations['Duration'].count()


# In[18]:


durations['Duration'].describe()


# In[19]:


durations.info()


# In[20]:


sns.boxplot(x='ZoneId',y='Duration',data=durations,hue='VehicleType',order=['LZ1','LZ2','LZ3','LZ4','LZ5','LZ6','LZ7','LZ8'])
plt.legend(loc='upper left',bbox_to_anchor=(1.02, 1))


# In[21]:


sns.boxplot(x='ZoneId',y='Duration',data=durations,hue='ProfessionalActivity',order=['LZ1','LZ2','LZ3','LZ4','LZ5','LZ6','LZ7','LZ8'])
plt.legend(loc='upper left',bbox_to_anchor=(1.02, 1))


# # **DB Cleaning**

# In[22]:


durations.drop(labels=['AreaFoundBy','PMR','OperationId','VehicleId','TimeLimit',
                        'Usage','VehicleAuthorized','ClosedBy','DriverId','Model'],axis=1,inplace=True)
durations.head()


# In[23]:


durations.tail()


# In[24]:


durations.info()


# # **Occupation analysis**

# In[25]:


help(hm.make_hills)


# In[ ]:


# Required inputs
# ---------------

# scenario name
#scenario = 'Occ_dur_LZ'

# Column name in trip_df corresponding to the bike rental time
#in_fld_name = 'TimeStart'

# Column name in trip_df corresponding to the bike return time
#out_fld_name = 'TimeFinish'

# Column name in trip_df corresponding to some categorical field for grouping
#cat_fld_name = ['ZoneId']

# Start and end times for the analysis. We'll just use data from 2015.
#start = '07/02/2018 00:00:00'
#end = '12/31/2019 22:00:00'

# Optional inputs
# ---------------

# Verbosity level of hillmaker messages
#verbose = 1

# Path to destination location for csv output files produced by hillmaker
#output = './content2'


# In[111]:


#hills_5 = hm.make_hills(scenario, duraciones, in_fld_name, out_fld_name, start, end, cat_fld_name, 
 #             bin_size_minutes=1,verbose=verbose)


# In[26]:


df_occ_LZ = pd.read_csv('Occ_dur_LZ_bydatetime_ZoneId_datetime.csv')


# In[27]:


df_occ_LZ


# In[28]:


df_occ_LZ['ZoneId'].isnull().value_counts()


# In[29]:


df_occ_LZ.dropna(axis=0,how='any',inplace=True)


# In[30]:


df_occ_LZ


# In[31]:


days_of_week = ['Monday','Tuesday','Wednesday','Thursday','Friday']
df_occ_hour = df_occ_LZ[(df_occ_LZ['dow_name'].isin(days_of_week)) & (df_occ_LZ['bin_of_day']>359) & (df_occ_LZ['bin_of_day']<1081)]
df_occ_hour['datetime'] = pd.to_datetime(df_occ_hour.loc[:,'datetime'])
df_occ_hour


# In[32]:


df_occ_hour.groupby(by='ZoneId').agg({'occupancy':'max'})


# In[33]:


df_occ_1 = df_occ_LZ[df_occ_LZ['arrivals']>0]
df_occ_1


# In[34]:


df_occ_1.info()


# In[35]:


df_occ_1['datetime'] = pd.to_datetime(df_occ_1.loc[:,'datetime'])


# In[36]:


df_occ_1.info()


# In[37]:


LZ_names = {'VIC-001':'LZ1','VIC-002':'LZ2', 
               'VIC-003':'LZ3','VIC-004':'LZ4',
               'VIC-005':'LZ5','VIC-006':'LZ6',
               'VIC-007':'LZ7','VIC-008':'LZ8'}


df_occ_1['ZoneId'] = df_occ_1['ZoneId'].map(LZ_names)


# In[38]:


df_occ_1


# # Association Analysis - GLZM

# ## Design matrix configuration

# In[39]:


durations


# In[40]:


df_validation=durations[durations['TimeStart']>'2019-09-25']


# In[41]:


df_validation.info()


# In[42]:


df_validation.dropna(inplace=True)


# In[43]:


df_validation


# In[44]:


vic_weather.head()


# In[45]:


durations['TimeWeather'] = durations['TimeStart'].dt.floor('H')


# In[46]:


durations.head()


# In[47]:


durations['TimeStart'] = durations['TimeStart'].dt.floor('Min')


# In[48]:


durations


# In[49]:


df = pd.merge(durations, df_occ_1,how='inner',left_on=['TimeStart','ZoneId'],right_on=['datetime','ZoneId'])


# In[50]:


df


# In[51]:


df.info()


# In[52]:


df['ZoneId'].isnull().value_counts()


# In[53]:


null = {'Emission':'Medium Emissions','ProfessionalActivity':'Unspecified'}
df.fillna(value=null,inplace=True)


# In[54]:


df.info()


# In[55]:


#Adding time variables
df['Month'] = df['TimeStart'].dt.month
df['QuarterStart'] = df['TimeStart'].dt.is_month_start
df['QuarterEnd'] = df['TimeStart'].dt.is_month_end

month_labels = {True:1,False:0}

df['QuarterStart'] = df['QuarterStart'].map(month_labels)
df['QuarterEnd'] = df['QuarterEnd'].map(month_labels)

df.rename(columns={'QuarterStart':'MonthStart','QuarterEnd':'MonthEnd'},inplace=True)

#Adding occupancy variables
LZ_capacity = {'LZ1':4,'LZ2':6,'LZ3':2,'LZ4':8,'LZ5':8,'LZ6':4,'LZ7':7,'LZ8':8}
df['Occupancy_Rate'] = df['ZoneId'].map(LZ_capacity)
df['Occupancy_Rate'] = df.loc[:,'occupancy']/df.loc[:,'Occupancy_Rate']

#Possible response variable transformation
df['logarithm_Dur'] = np.log10(df['Duration'])


# In[56]:


df


# In[57]:


df.corr()


# In[58]:


vic_weather.columns


# In[59]:


vic_weather.rename(columns={'date_time':'TimeWeather'},inplace=True)


# In[60]:


vic_weather.describe()


# In[61]:


vic_weather.drop(labels=['totalSnow_cm','uvIndex','moon_illumination','DewPointC','HeatIndexC',
                        'WindChillC','WindGustKmph','pressure','visibility','winddirDegree',
                        'moonrise','moonset','sunrise','sunset','tempC','location'],axis=1,inplace=True)
vic_weather.head()


# In[62]:


vic_weather['TimeWeather'] = pd.to_datetime(vic_weather.loc[:,'TimeWeather'])

vic_weather.info()


# In[63]:


vic_weather.set_index('TimeWeather')


# In[64]:


vic_weather.info()


# In[117]:


df_w = pd.merge(df,vic_weather, left_on='TimeWeather',right_on='TimeWeather',how='left',sort=False)


# In[118]:


df_w['hour']= df_w['TimeStart'].dt.hour
df_w.hour = df_w.hour.astype('object')


# In[119]:


pd.set_option("display.max_columns", None)
df_w.head()


# In[120]:


df_w.info()


# In[121]:


df_w.columns


# In[ ]:


#https://dius.com.au/2017/08/03/using-statsmodels-glms-to-model-beverage-consumption/
#https://statmath.wu.ac.at/courses/heather_turner/index.html

model_full3 = 'Duration ~ C(hour) + C(ProfessionalActivity, Treatment(reference="Unspecified")) + C(ZoneId) + C(VehicleType) + C(Emission) + FeelsLikeC + precipMM'
response2, predictors2 = dmatrices(model_full3, df_w, return_type='dataframe')
lmfull3    = sm.GLM(response2,predictors2, family=sm.families.Gamma(link=sm.genmod.families.links.log)).fit()
print(lmfull3.summary())


# In[ ]:


lmfull3.aic


# In[93]:


lmfull3.pearson_chi2 / lmfull3.df_resid


# # Queueing Modelling

# In[73]:


sns.set(style="whitegrid", font_scale=1)


# In[74]:


data = df['Duration'].values

f = Fitter(data,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
f.fit()
f.summary()


# In[75]:


#https://fitter.readthedocs.io/en/latest/ 
f = Fitter(data)
f.fit()
# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()


# In[76]:


f.fitted_param['johnsonsb']


# ## X,y formating for Clustering

# In[77]:


df.info()


# In[78]:


df.head()


# In[79]:


X_q_categoric = df[['VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','Month',]].values
X_q_numeric = df[['bin_of_day','Occupancy_Rate']].values
y_q = df['Duration'].values
y_q_round=np.rint(y_q)

print('Matrix shape - X categoric: ',X_q_categoric.shape)
print('Matrix shape - X numeric: ',X_q_numeric.shape)
print('Array shape - y: ',y_q.shape)
print('Array shape - y_round: ',y_q_round.shape)


# In[81]:


for i in ['VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','Month']:
  print(i,'\n',df[i].value_counts())


# In[83]:


enc = OneHotEncoder(sparse=False) 
X_q_categoric_onehot = enc.fit_transform(X_q_categoric)
print(X_q_categoric_onehot.shape)
print(type(X_q_categoric_onehot))


# In[91]:


scaler = MinMaxScaler(feature_range=(0, 1))  
X_q_numeric_minmax = scaler.fit_transform(X_q_numeric)


# In[92]:


X_q_full = np.concatenate((X_q_numeric_minmax, X_q_categoric_onehot),
                        axis=1)
print(X_q_full.shape)


# ## Dimensionality Reduction

# In[93]:


pca = PCA() 
transf = pca.fit_transform(X_q_full)

varianza_expl = pca.explained_variance_ratio_

print(varianza_expl)


# In[87]:


def cumulative_explained_variance_plot(expl_variance):

  cum_var_exp = np.cumsum(expl_variance)

  plt.figure(dpi = 100, figsize = (8, 6))
  plt.title('Cumulative explained variance plot VS Number of Principal Components', 
            fontdict= dict(family ='serif', size = 16))
  plt.xlabel('Number of Principal Components',
             fontdict= dict(family ='serif', size = 14))
  plt.ylabel('Cumulative explained variance',
             fontdict= dict(family ='serif', size = 14))  

  nc = np.arange(1, expl_variance.shape[0] + 1)

  plt.plot(nc, cum_var_exp, '--r')
  plt.plot(nc, cum_var_exp, 'c*', ms = 5)   
  plt.show()


# In[94]:


cumulative_explained_variance_plot(varianza_expl)


# In[95]:


np.cumsum(varianza_expl)


# In[96]:


pca = PCA(n_components= 22)
X_q_pca = pca.fit_transform(X_q_full)
pca.components_


# In[97]:


X_q_pca.shape


# ## Clustering for durations in queueing analysis

# In[98]:


X_q_pca


# In[100]:


sum_sq_d = []
K = range(1,11)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_q_pca)
    sum_sq_d.append(km.inertia_)
    
plt.figure(figsize=(8,6))

plt.plot(K, sum_sq_d, 'rx-.')

plt.xlabel('# Clusters, k', fontsize=12)
plt.xticks(range(1,11), fontsize=12)

plt.ylabel('Inertia', fontsize=12)
plt.xticks(fontsize=12)

plt.title('K vs Inertia', fontsize=16)

plt.show()


# In[101]:


def plot_metric(K, scores, metric_name):
  plt.figure(dpi=110, figsize=(9, 5))
  plt.plot(K, scores, 'bx-')
  plt.xticks(K); plt.xlabel('$k$', fontdict=dict(family = 'serif', size = 14));  plt.ylabel(metric_name, fontdict=dict(family = 'serif', size = 14));
  plt.title(f'K vs {metric_name}', fontdict=dict(family = 'serif', size = 18))
  plt.show()


# In[102]:


from sklearn.metrics import silhouette_score

silhouette_scores = []

K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_q_pca)
    y_cl = km.predict(X_q_pca)
    silhouette_scores.append(silhouette_score(X_q_pca, y_cl))

plot_metric(K, silhouette_scores, 'Coeficiente de silueta')


# In[103]:


k = 5

kmeans = KMeans(n_clusters=k, init='k-means++')
kmeans.fit(X_q_pca)

labels = kmeans.predict(X_q_pca)

centroids = kmeans.cluster_centers_
df['Classification_Kmeans'] = pd.Series(labels+1, index=df.index)


# In[104]:


sns.violinplot(x='Classification_Kmeans',y='Duration', data=df[df['ZoneId']=='LZ8'])


# In[105]:


sns.violinplot(df.TimeStart.dt.hour,y='Duration',order=[6,7,8,9,10,11,12,13,14,15,16,16,18], data=df[(df['ZoneId']=='LZ4')&(df['Classification_Kmeans']==2)])


# # Clustering without LZ category

# In[106]:


X_q_categoric_LZ = df[['VehicleType','Emission','ProfessionalActivity','day_of_week','Month']].values
X_q_numeric_LZ = df[['bin_of_day','Occupancy_Rate']].values
y_q_LZ = df['Duration'].values
y_q_round_LZ=np.rint(y_q_LZ)

print('Matrix shape - X categoric: ',X_q_categoric_LZ.shape)
print('Matrix shape - X numeric: ',X_q_numeric_LZ.shape)
print('Array shape - y: ',y_q_LZ.shape)
print('Array shape - y_round: ',y_q_round_LZ.shape)


# In[108]:


X_q_categoric_onehot_LZ = enc.fit_transform(X_q_categoric_LZ)
print(X_q_categoric_onehot_LZ.shape)
print(type(X_q_categoric_onehot_LZ))


# In[109]:


X_q_numeric_minmax_LZ = scaler.fit_transform(X_q_numeric_LZ)


# In[110]:


X_q_full_LZ = np.concatenate((X_q_numeric_minmax_LZ, X_q_categoric_onehot_LZ),
                        axis=1)
print(X_q_full_LZ.shape)


# In[114]:


pca_q = PCA()
transf2 = pca_q.fit_transform(X_q_full_LZ)

varianza_expl2 = pca_q.explained_variance_ratio_

print(varianza_expl2)


# In[115]:


cumulative_explained_variance_plot(varianza_expl2)


# In[116]:


np.cumsum(varianza_expl2)


# In[117]:


pca_q = PCA(n_components= 16)
X_q_pca_LZ = pca_q.fit_transform(X_q_full_LZ)
pca_q.components_


# In[118]:


sum_sq_d = []
K = range(1,11)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_q_pca_LZ)
    sum_sq_d.append(km.inertia_)
    
plt.figure(figsize=(8,6))

plt.plot(K, sum_sq_d, 'rx-.')

plt.xlabel('# Clusters, k', fontsize=12)
plt.xticks(range(1,11), fontsize=12)

plt.ylabel('Inercia', fontsize=12)
plt.xticks(fontsize=12)

plt.title('K vs Inertia', fontsize=16)

plt.show()


# In[119]:


silhouette_scores = []

K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_q_pca_LZ)
    y_cl_LZ = km.predict(X_q_pca_LZ)
    silhouette_scores.append(silhouette_score(X_q_pca_LZ, y_cl_LZ))

plot_metric(K, silhouette_scores, 'Coeficiente de silueta')


# In[120]:


k = 5

kmeans = KMeans(n_clusters=k, init='k-means++')
kmeans.fit(X_q_pca_LZ)

labels = kmeans.predict(X_q_pca_LZ)

centroids = kmeans.cluster_centers_
df['Classification_Kmeans_q_LZ'] = pd.Series(labels+1, index=df.index)


# In[142]:


df[df['ZoneId']=='LZ4'].Classification_Kmeans_q_LZ.value_counts()/len(df[df['ZoneId']=='LZ4'])


# In[129]:


sns.set(style="whitegrid", font_scale=1)


# In[134]:


sns.violinplot(df.TimeStart.dt.hour,y='Duration',order=[6,7,8,9,10,11,12,13,14,15,16,17,18], data=df[(df['ZoneId']=='LZ4')&(df['Classification_Kmeans_q_LZ']==5)])


# In[159]:


data_LZ_cl = df[(df['ZoneId']=='LZ4')&(df['Classification_Kmeans_q_LZ']==5)&(df['TimeStart'].dt.hour==14)]
data2 = data_LZ_cl['Duration'].values


# In[160]:


f = Fitter(data2)
f.fit()
# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()


# In[161]:


f = Fitter(data2,
           distributions=['exponnorm'])
f.fit()
f.summary()


# In[158]:


f.fitted_param['exponnorm']


# # MACHINE LEARNING MODELLING

# ## Data Partitioning

# In[96]:


df_w.info()


# In[97]:


df_w.columns


# In[98]:


X_categoric = df_w[['VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','Month','hour']].values
X_numeric = df_w[['FeelsLikeC', 'precipMM']].values
y = df_w['Duration'].values
y_round=np.rint(y)

print('Matrix shape - X categoric: ',X_categoric.shape)
print('Matrix shape - X numeric: ',X_numeric.shape)
print('Array shape - y: ',y.shape)
print('Array shape - y_round: ',y_round.shape)


# In[99]:


enc = OneHotEncoder(sparse=False) 
X_categoric_onehot = enc.fit_transform(X_categoric)
print(X_categoric_onehot.shape)
print(type(X_categoric_onehot))


# In[100]:


scaler = MinMaxScaler(feature_range=(0, 1))  
X_numeric_minmax = scaler.fit_transform(X_numeric)


# In[101]:


X_full = np.concatenate((X_numeric_minmax, X_categoric_onehot),
                        axis=1)
print(X_full.shape)


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X_full, y_round, test_size=0.2)
print("Training matrix shape: ",X_train.shape,"Test matrix shape: ", X_test.shape)
print("Training durations: ",y_train.shape,"Test durations: ", y_test.shape)


# ## Linear Regression (Base Model)

# In[157]:


reg_full = linear_model.LinearRegression()
reg_full.fit(X_train, y_train)


# In[158]:


y_pred_train_LR = reg_full.predict(X_train)
y_pred_test_LR = reg_full.predict(X_test)


# In[159]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_LR))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_LR))


# In[160]:


print('MSE training',mean_squared_error(y_train, y_pred_train_LR))
print('MSE testing',mean_squared_error(y_test, y_pred_test_LR))


# In[161]:


mean_squared_error(y_train, y_pred_train_LR)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_LR)))

mean_squared_error(y_test, y_pred_test_LR)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_LR)))


# In[162]:


print('R2 training',r2_score(y_train, y_pred_train_LR))
print('R2 testing', r2_score(y_test, y_pred_test_LR))


# In[113]:


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


# In[164]:


print('SMAPE training',smape(y_pred_train_LR, y_train),
      'SMAPE testing',smape( y_pred_test_LR,y_test))


# In[165]:


scores = cross_val_score(reg_full, 
                         X_train, 
                         y_train, scoring = 'r2', 
                         cv = 5)


# In[166]:


print(f'Splits R2:\n{list(scores)}')
print(f'Average R2: {np.mean(scores)}')


# In[105]:


std_slc0 = StandardScaler()
pca0 = PCA()
lin_reg1 = linear_model.LinearRegression()

pipe0 = Pipeline(steps=[('std_slc', std_slc0),
                           ('pca', pca0),
                           ('lin_reg', lin_reg1)])

n_components = list(range(1,X_full.shape[1]+1,1))
    

parameters0 = dict(pca__n_components=n_components)


# In[106]:


lr_RS = RandomizedSearchCV(pipe0, parameters0,n_jobs=-1)


# In[124]:


y_train1=stats.yeojohnson(y_train+0.001)


# In[125]:


y_test1=stats.yeojohnson(y_test+0.001)


# In[126]:


len(y_train1)


# In[107]:


lr_RS.fit(X_train, y_train)


# In[108]:


y_pred_train_LRRS = lr_RS.predict(X_train)
y_pred_test_LRRS = lr_RS.predict(X_test)


# In[109]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_LRRS))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_LRRS))


# In[110]:


print('MSE training',mean_squared_error(y_train, y_pred_train_LRRS))
print('MSE testing',mean_squared_error(y_test, y_pred_test_LRRS))


# In[111]:


mean_squared_error(y_train, y_pred_train_LRRS)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_LRRS)))

mean_squared_error(y_test, y_pred_test_LRRS)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_LRRS)))


# In[112]:


mean_squared_error(y_train, y_pred_train_LRRS)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_LRRS)))

mean_squared_error(y_test, y_pred_test_LRRS)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_LRRS)))


# In[113]:


print('R2 training',r2_score(y_train, y_pred_train_LRRS))
print('R2 testing', r2_score(y_test, y_pred_test_LRRS))


# In[116]:


print('SMAPE training',smape(y_pred_train_LRRS, y_train),
      'SMAPE testing',smape( y_pred_test_LRRS,y_test))


# In[117]:


lr_RS.best_params_


# ## Decision Trees

# In[118]:


#https://www.dezyre.com/recipes/optimize-hyper-parameters-of-decisiontree-model-using-grid-search-in-python
#https://www.nbshare.io/notebook/312837011/Decision-Tree-Regression-With-Hyper-Parameter-Tuning-In-Python/

std_slc2 = StandardScaler()
pca2 = PCA()
dec_tree2 = DecisionTreeRegressor()

pipe2 = Pipeline(steps=[('std_slc', std_slc2),
                           ('pca', pca2),
                           ('dec_tree', dec_tree2)])

n_components = list(range(1,X_full.shape[1]+1,1))
    
splitter = ["best","random"]
max_depth = [2,4,6,8,12,16,24,30]
min_samples_leaf = [1,2,3,4,5,6,7,8,9,10]
min_weight_fraction_leaf = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
max_features = ["auto","log2","sqrt",None]
max_leaf_nodes = [None,10,20,30,40,50,60,70,80,90]

parameters = dict(pca__n_components=n_components,
                      dec_tree__splitter=splitter,
                      dec_tree__max_depth=max_depth,
                      dec_tree__min_samples_leaf = min_samples_leaf,
                      dec_tree__min_weight_fraction_leaf = min_weight_fraction_leaf,
                      dec_tree__max_features = max_features,
                      dec_tree__max_leaf_nodes = max_leaf_nodes)


# In[119]:


dt_RS = RandomizedSearchCV(pipe2,parameters,n_jobs=-1)


# In[120]:


dt_RS.fit(X_train,y_train)


# In[121]:


best_dt_RS = dt_RS.best_estimator_


# In[122]:


y_pred_train_DTRS = best_dt_RS.predict(X_train)
y_pred_test_DTRS = best_dt_RS.predict(X_test)


# In[123]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_DTRS))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_DTRS))


# In[124]:


print('MSE training',mean_squared_error(y_train, y_pred_train_DTRS))
print('MSE testing',mean_squared_error(y_test, y_pred_test_DTRS))


# In[125]:


mean_squared_error(y_train, y_pred_train_DTRS)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_DTRS)))

mean_squared_error(y_test, y_pred_test_DTRS)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_DTRS)))


# In[126]:


print('R2 training',r2_score(y_train, y_pred_train_DTRS))
print('R2 testing', r2_score(y_test, y_pred_test_DTRS))


# In[127]:


print('SMAPE training',smape(y_pred_train_DTRS, y_train),
      'SMAPE testing',smape( y_pred_test_DTRS,y_test))


# In[128]:


dt_RS.best_params_


# ## Random Forest

# In[189]:


randomF = RandomForestRegressor()
randomF.fit(X_train, y_train)


# In[190]:


y_pred_train_RF = randomF.predict(X_train)
y_pred_test_RF = randomF.predict(X_test)


# In[191]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_RF))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_RF))


# In[192]:


print('MSE training',mean_squared_error(y_train, y_pred_train_RF))
print('MSE testing',mean_squared_error(y_test, y_pred_test_RF))


# In[193]:


mean_squared_error(y_train, y_pred_train_RF)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_RF)))

mean_squared_error(y_test, y_pred_test_RF)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_RF)))


# In[194]:


print('R2 training',r2_score(y_train, y_pred_train_RF))
print('R2 testing', r2_score(y_test, y_pred_test_RF))


# In[129]:


std_slc3 = StandardScaler()
pca3 = PCA()
random_forest2 = RandomForestRegressor()

pipe3 = Pipeline(steps=[('std_slc', std_slc3),
                           ('pca', pca3),
                           ('rand_forest', random_forest2)])

n_components = list(range(1,X_full.shape[1]+1,1))
    
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 150, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 24, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

parameters = dict(pca__n_components=n_components,
                      rand_forest__n_estimators=n_estimators,
                      rand_forest__max_features=max_features,
                      rand_forest__max_depth = max_depth,
                      rand_forest__min_samples_split =min_samples_split,
                      rand_forest__min_samples_leaf = min_samples_leaf,
                      rand_forest__bootstrap = bootstrap)


# In[130]:


rf_RS = RandomizedSearchCV(pipe3, parameters, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)


# In[131]:


rf_RS.fit(X_train, y_train)


# In[134]:


rf_RS.best_params_


# In[135]:


best_random = rf_RS.best_estimator_


# In[136]:


y_pred_train_RFRS = best_random.predict(X_train)
y_pred_test_RFRS = best_random.predict(X_test)


# In[137]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_RFRS))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_RFRS))


# In[138]:


print('MSE training',mean_squared_error(y_train, y_pred_train_RFRS))
print('MSE testing',mean_squared_error(y_test, y_pred_test_RFRS))


# In[139]:


mean_squared_error(y_train, y_pred_train_RFRS)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_RFRS)))

mean_squared_error(y_test, y_pred_test_RFRS)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_RFRS)))


# In[140]:


print('R2 training',r2_score(y_train, y_pred_train_RFRS))
print('R2 testing', r2_score(y_test, y_pred_test_RFRS))


# In[141]:


print('SMAPE training',smape(y_pred_train_RFRS, y_train),
      'SMAPE testing',smape( y_pred_test_RFRS,y_test))


# In[142]:


rf_RS.best_params_


# ## **XGBoost**

# In[143]:


import sys
print(sys.base_prefix)


# In[144]:


conda install -c conda-forge py-xgboost


# In[145]:


import xgboost as xgb # XGBoost library

def XGBmodel(x_train, x_val, y_train, y_val):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_val = xgb.DMatrix(x_val, label=y_val)
    model=xgb.train(params={'objective':'reg:gamma','eval_metric':'rmse'},
                    dtrain=matrix_train,
                    num_boost_round=500, 
                    early_stopping_rounds=90,
                    evals=[(matrix_val,'validation')],)
    return model


# In[146]:


model=XGBmodel(X_train, X_test, y_train, y_test)


# In[147]:


y_pred_train_xg = model.predict(xgb.DMatrix(X_train), ntree_limit = model.best_ntree_limit)
y_pred_test_xg = model.predict(xgb.DMatrix(X_test), ntree_limit = model.best_ntree_limit)


# In[148]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_xg))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_xg))


# In[149]:


print('MSE training',mean_squared_error(y_train, y_pred_train_xg))
print('MSE testing',mean_squared_error(y_test, y_pred_test_xg))


# In[150]:


mean_squared_error(y_train, y_pred_train_xg)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_xg)))

mean_squared_error(y_test, y_pred_test_xg)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_xg)))


# In[151]:


print('R2 training',r2_score(y_train, y_pred_train_xg))
print('R2 testing', r2_score(y_test, y_pred_test_xg))


# In[152]:


print('SMAPE training',smape(y_pred_train_xg, y_train),
      'SMAPE testing',smape( y_pred_test_xg,y_test))


# In[153]:


xgb.XGBRegressor()


# In[154]:


xgb2 = xgb.XGBRegressor()
std_slc4 = StandardScaler()
pca4 = PCA()

pipe4 = Pipeline(steps=[('std_slc', std_slc4),
                           ('pca', pca4),
                           ('xgb', xgb2)])

n_components = list(range(1,X_full.shape[1]+1,1))

learning_rate = [0.17,0.2,0.21,0.25,0.3,0.35] 
min_child_weight = [3,4, 5,7, 9] 
gamma = [1,5,7,8,9,10]
subsample = [.5,.6,.7, .8,.9]
colsample_bytree = [.8, .9, 1] 
max_depth = [2,3,4,5,6] 

parameters = dict(pca__n_components=n_components,
                      xgb__learning_rate=learning_rate,
                      xgb__min_child_weight=min_child_weight,
                      xgb__gamma = gamma,
                      xgb__subsample =subsample,
                      xgb__colsample_bytree = colsample_bytree,
                      xgb__max_depth = max_depth)


# In[155]:


xgb_RS = RandomizedSearchCV(pipe4, parameters, verbose=2, n_jobs = -1)


# In[156]:


xgb_RS.fit(X_train, y_train)


# In[157]:


best_xgb_RS = xgb_RS.best_estimator_


# In[158]:


xgb_RS.best_params_


# In[159]:


y_pred_train_xgRS = best_xgb_RS.predict(X_train)
y_pred_test_xgRS = best_xgb_RS.predict(X_test)


# In[160]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_xgRS))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_xgRS))


# In[161]:


print('MSE training',mean_squared_error(y_train, y_pred_train_xgRS))
print('MSE testing',mean_squared_error(y_test, y_pred_test_xgRS))


# In[162]:


mean_squared_error(y_train, y_pred_train_xgRS)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_xgRS)))

mean_squared_error(y_test, y_pred_test_xgRS)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_xgRS)))


# In[251]:


print('R2 training',r2_score(y_train, y_pred_train_xgRS))
print('R2 testing', r2_score(y_test, y_pred_test_xgRS))


# In[163]:


print('SMAPE training',smape( y_pred_train_xgRS, y_train),
      'SMAPE testing',smape( y_pred_test_xgRS,y_test))


# ## CatBoost

# https://towardsdatascience.com/catboost-regression-in-6-minutes-3487f3e5b329

# In[77]:


pip install catboost


# In[93]:


import catboost as cb


# In[477]:


df_X = df[{'VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','hour','Month'}]


# In[478]:


df_y= df['Duration'].values


# In[78]:


df_w.columns


# In[79]:


features = ['VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','hour','Month','FeelsLikeC',
           'precipMM']
cat_features = ['VehicleType','Emission','ProfessionalActivity','ZoneId','day_of_week','hour','Month']
target = 'Duration'
df_CB = df_w[features + [target]]
df_train, df_test = train_test_split(df_CB)


# In[125]:


df_train


# In[128]:


from catboost import CatBoostRegressor, Pool

train_pool = Pool(df_train[features], label=df_train[target],
                  cat_features=cat_features)
test_pool = Pool(df_test[features], label=df_test[target],
                 cat_features=cat_features)


# In[129]:


cb_tweedie = CatBoostRegressor(loss_function='Tweedie:variance_power=1.9', n_estimators=500, silent=True)
cb_tweedie.fit(train_pool, eval_set=test_pool)


# In[143]:


cb_final = CatBoostRegressor(nan_mode = 'Min', eval_metric='RMSE',iterations= 200,
 sampling_frequency= 'PerTree',
 leaf_estimation_method= 'Newton',
 grow_policy= 'SymmetricTree',
 penalties_coefficient= 1,
 boosting_type= 'Plain',
 model_shrink_mode= 'Constant',
 feature_border_type= 'GreedyLogSum',
 #bayesian_matrix_reg= 0.10000000149011612,
 l2_leaf_reg= 6,
 random_strength= 1,
 rsm= 1,
 boost_from_average= True,
 model_size_reg= 0.5,
 #pool_metainfo_options= {'tags': {}},
 subsample= 0.800000011920929,
 use_best_model= False,
 random_seed= 0,
 depth= 10,
 posterior_sampling= False,
 border_count= 254,
 #classes_count= 0,
 #auto_class_weights= 'None',
 sparse_features_conflict_fraction= 0,
 leaf_estimation_backtracking= 'AnyImprovement',
 best_model_min_trees= 1,
 model_shrink_rate= 0,
 min_data_in_leaf= 1,
 loss_function= 'RMSE',
 learning_rate= 0.10000000149011612,
 score_function= 'Cosine',
 task_type= 'CPU',
 leaf_estimation_iterations= 1,
 bootstrap_type= 'MVS',
 max_leaves= 1024)


# In[144]:


cb_final.fit(train_pool, eval_set=test_pool)


# In[145]:


CB_pred_train0 = cb_final.predict(train_pool)
CB_pred_test0 = cb_final.predict(test_pool)


# In[146]:


test_pool


# In[147]:


from catboost.utils import eval_metric

eval_metric(df_test[target].to_numpy(), CB_pred_test0, 'MAE')


# In[148]:


eval_metric(df_test[target].to_numpy(), CB_pred_test0, 'RMSE')


# In[149]:


eval_metric(df_test[target].to_numpy(), CB_pred_test0, 'R2')


# In[150]:


eval_metric(df_test[target].to_numpy(), CB_pred_test0, 'SMAPE')


# In[151]:


feature_importance = cb_tweedie.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(df_train.columns)[sorted_idx])
plt.title('Feature Importance')


# In[137]:


sns.set(style="whitegrid", font_scale=2)


# ### Random search

# In[82]:


from scipy import stats

params_distribution = {
    'learning_rate': stats.uniform(0.01, 0.1),
    'depth': list(range(3, 10)),
    'l2_leaf_reg': stats.uniform(1, 10),
    'boosting_type': ['Ordered', 'Plain'],
    
}

cb_tweedie_model_rs = CatBoostRegressor(loss_function='RMSE')
cb_tweedie_results_rs = cb_tweedie_model_rs.randomized_search(
    params_distribution, 
    train_pool, 
    n_iter=20, 
    verbose=5, 
    partition_random_seed=123)


# In[83]:


cb_tweedie_results_rs['params']


# In[84]:


cb_tweedie_rs = CatBoostRegressor(depth= 8, learning_rate = 0.05884606200007537, l2_leaf_reg = 9.141341319053957,
 boosting_type= 'Ordered',loss_function='RMSE', n_estimators=500, silent=True)
cb_tweedie_rs.fit(train_pool, eval_set=test_pool)


# In[85]:


CB_pred_train_rs = cb_tweedie_rs.predict(train_pool)
CB_pred_test_rs = cb_tweedie_rs.predict(test_pool)


# In[88]:


eval_metric(df_test[target].to_numpy(), CB_pred_test_rs, 'MAE')


# In[89]:


eval_metric(df_test[target].to_numpy(), CB_pred_test_rs, 'RMSE')


# In[90]:


eval_metric(df_test[target].to_numpy(), CB_pred_test_rs, 'R2')


# In[91]:


eval_metric(df_test[target].to_numpy(), CB_pred_test_rs, 'SMAPE')


# ### Grid search

# In[103]:


train_dataset = cb.Pool(X_train, y_train) 
test_dataset = cb.Pool(X_test, y_test)


# In[104]:


CB = cb.CatBoostRegressor(loss_function='RMSE')


# In[105]:


grid = {'iterations': [200, 350, 500],
        'learning_rate': [0.1, 0.2, 0.3],
        'depth': [8, 10, 12, 14],
        'l2_leaf_reg': [3, 4.5, 6, 7.5]}
CB.grid_search(grid, train_dataset)


# In[106]:


CB.get_all_params()''


# In[107]:


CB_pred_train = CB.predict(X_train)
CB_pred_test = CB.predict(X_test)


# In[108]:


print('MAE training',mean_absolute_error(y_train, CB_pred_train))
print('MAE testing',mean_absolute_error(y_test, CB_pred_test))


# In[109]:


print('MSE training',mean_squared_error(y_train, CB_pred_train))
print('MSE testing',mean_squared_error(y_test, CB_pred_test))


# In[110]:


mean_squared_error(y_train, CB_pred_train)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, CB_pred_train)))

mean_squared_error(y_test, CB_pred_test)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, CB_pred_test)))


# In[111]:


print('R2 training',r2_score(y_train, CB_pred_train))
print('R2 testing', r2_score(y_test, CB_pred_test))


# In[114]:


print('SMAPE training',smape(CB_pred_train, y_train),
      'SMAPE testing',smape(CB_pred_test,y_test))


# In[115]:


feature_importance = CB.get_feature_importance(type='PredictionValuesChange',prettified=False)
feature_importance


# In[ ]:


feature_importance = CB.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(df_train.columns)[sorted_idx])
plt.title('Feature Importance')


# In[ ]:


X_train


# In[ ]:


pd.DataFrame({'feature_importance': CB.get_feature_importance(CB_pred_train), 
              'feature_names': CB_pred_train.columns}).sort_values(by=['feature_importance'], ascending=False)
                                                   


# In[ ]:


object_importance = CB.get_object_importance(train_dataset,
                      train_dataset,
                      top_size=-1,
                      type='Average',
                      update_method='SinglePoint',
                      importance_values_sign='All',
                      thread_count=-1,
                      verbose=False,
                      log_cout=sys.stdout,
                      log_cerr=sys.stderr)
object_importance


# In[123]:


feature_importance2 = CB.get_feature_importance(type='Interaction',prettified=True)
feature_importance2.head(40)


# In[ ]:


X_categoric[61521]


# In[284]:


X_full[61521]


# ## Neural Networks

# In[152]:


from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

params = {    
      'hidden_layer_sizes' : [(80,), (100,), (120,), (180,) ], # Some architectures.
      'activation' : ['logistic', 'tanh', 'relu'],           # Activation functions.
      'learning_rate':['adaptative','invscaling','constant'],
    'max_iter':[200,400]
 }

gsearch_nn = GridSearchCV(estimator = MLPRegressor(solver = 'adam', #Model to be explored.          
                                                random_state=1234,
                                                max_iter= 2000,
                                                n_iter_no_change=50, 
                                                validation_fraction=0.2), 
                        param_grid = params, 
                        verbose = 3)

gsearch_nn.fit(X_train, y_train)


# In[153]:


y_pred_train_nn = gsearch_nn.predict(X_train)
y_pred_test_nn = gsearch_nn.predict(X_test)


# In[154]:


gsearch_nn.best_params_


# In[155]:


print('MAE training',mean_absolute_error(y_train, y_pred_train_nn))
print('MAE testing',mean_absolute_error(y_test, y_pred_test_nn))


# In[156]:


print('MSE training',mean_squared_error(y_train, y_pred_train_nn))
print('MSE testing',mean_squared_error(y_test, y_pred_test_nn))


# In[157]:


mean_squared_error(y_train, y_pred_train_nn)
 
print('RMSE training', np.sqrt(mean_squared_error(y_train, y_pred_train_nn)))

mean_squared_error(y_test, y_pred_test_nn)
 
print('RMSE testing',np.sqrt(mean_squared_error(y_test, y_pred_test_nn)))


# In[158]:


print('R2 training',r2_score(y_train, y_pred_train_nn))
print('R2 testing', r2_score(y_test, y_pred_test_nn))


# In[159]:


print('SMAPE training',smape(y_pred_train_nn, y_train),
      'SMAPE testing',smape(y_pred_test_nn,y_test))

