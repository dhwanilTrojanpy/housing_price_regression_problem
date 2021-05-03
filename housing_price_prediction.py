#!/usr/bin/env python
# coding: utf-8

# 
# # Steps to follow
# # 1) EDA
# # 2) Split dataset
# # 3) One hot encoding
# # 4) Missing value imputation
# # 5) Normalization or Standardization
# # 6) Remove Outliers
# # 7) PCA
# # 8) Feature Selection
# 

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
import xgboost as XGB


# In[5]:


Train = pd.read_csv(r"C:\Users\A\Desktop\kaggle\train.csv")
Test= pd.read_csv(r"C:\Users\A\Desktop\kaggle\test.csv")
#Train.head()


# In[6]:


Train = Train.drop("Id",axis = 1)   #### ID is an unneessary column


# # EDA

# In[7]:


Train.info()


# In[ ]:





# In[8]:


Train.shape


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
corrmat = Train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,square=True);


# In[10]:


corr = Train.corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(Train[highest_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[11]:


print(corr.index[abs(corr["SalePrice"])>0.5])


# In[12]:


y_train = Train['SalePrice']
test_id = Test['Id']
dataset = pd.concat([Train, Test], axis=0, sort=False)
dataset = dataset.drop(['SalePrice'], axis=1)


# ## impute columns values with repsect to mean and median values
# ## the value which is near to 25% is imputed here

# In[13]:


count = 0
null_value_columns=[]
for i in dataset.columns:   #### find out all the columns with null values
    #print(i + ":", end = " ")
    #print(dataset[i].isna().sum())
    if dataset[i].isna().sum() > 0:
        null_value_columns.append(i)
        count +=1
#print("total columns with null values : " + str(count))    


# In[14]:


print("columns with null values :", null_value_columns)
print("\n")
highest_corr_features_list=list(highest_corr_features)
print("columns of high correlation :", highest_corr_features_list)
print("\n")

unwanted_columns=[]
unwanted_count =0 
for columns in null_value_columns:
    if columns not in highest_corr_features_list:
        unwanted_columns.append(columns)
        unwanted_count +=1
print("unwanted columns:", unwanted_columns )  




### drop features that are correlated with each other
#mat = data.corr()
#list_of _fearture =[i for i in mat]
#set_of_drop_features= set()
#for i in range(list(set_of_drop_features)):
 #   for j in range(i+1,len(list_of_features)):
  #      feature1=list_of_features[i]
   #     feature2=list_of_features[j]
    #    if abs(mat[feature1][feature2]) > 0.7:
     #       set_of_drop_features.add(feature1)
#data= data.drop(set_of_drop_features,axis =1)

###Drop features that are not correlated with output

##not_correlated =[column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.0045]
###data = data.drop(set_of_drop_features, axis=1)


# In[15]:


#### delete unwanted columns because correlation is less than 0.5

for columns in unwanted_columns:
    dataset = dataset.drop([columns], axis=1)


# In[16]:


dataset


# In[17]:


count = 0
for i in dataset.columns:   #### find out all the columns with null values
    print(i + ":", end = " ")
    print(dataset[i].isna().sum())
    if dataset[i].isna().sum() > 0:
        count +=1
#print("total columns with null values : " + str(count))


# In[18]:


### impute all numerical and categorical values in one step

dataset = dataset.apply(lambda x:x.fillna(x.mean()) 
                    if x.dtype == "float" or x.dtype =="int64" 
                    else x.fillna(x.value_counts().index[0]))


#Train["LotFrontage"] = Train["LotFrontage"].fillna(Train["LotFrontage"].median())        
    


# In[16]:


#Train["Alley"].unique()
#Train["Alley"] = Train["Alley"].fillna(Train["Alley"].value_counts().index[0])
#Train["MasVnrType"] = Train["MasVnrType"].fillna(Train["MasVnrType"].value_counts().index[0])
#Train["MasVnrType"].unique()


# In[17]:


#Train["MasVnrArea"] = Train["MasVnrArea"].fillna(Train["MasVnrArea"].median())        
#train["MasVnrArea"].isna().sum()


# In[18]:


#print(Train["BsmtQual"].unique())
#print(Train["BsmtCond"].unique())
#print(Train["BsmtExposure"].unique()) 
#print(Train["BsmtFinType1"].unique())
#print(Train["BsmtFinType2"].unique())
#print(Train["Electrical"].unique())

#Train["BsmtQual"] = Train["BsmtQual"].fillna(Train["BsmtQual"].value_counts().index[0])
#Train["BsmtCond"] = Train["BsmtCond"].fillna(Train["BsmtCond"].value_counts().index[0])
#Train["BsmtExposure"] = Train["BsmtExposure"].fillna(Train["BsmtExposure"].value_counts().index[0])
#Train["BsmtFinType1"] = Train["BsmtFinType1"].fillna(Train["BsmtFinType1"].value_counts().index[0])
#Train["BsmtFinType2"] = Train["BsmtFinType2"].fillna(Train["BsmtFinType2"].value_counts().index[0]) 
#Train["Electrical"] = Train["Electrical"].fillna(Train["Electrical"].value_counts().index[0]) 


# In[19]:


#print(Train["FireplaceQu"].unique())
#print(Train["GarageType"].unique())
#print(Train["GarageYrBlt"].unique())
#print(Train["GarageFinish"].unique())
#print(Train["GarageQual"].unique())
#print(Train["GarageCond"].unique())

#Train["FireplaceQu"] = Train["FireplaceQu"].fillna(Train["FireplaceQu"].value_counts().index[0])
#Train["GarageType"] = Train["GarageType"].fillna(Train["GarageType"].value_counts().index[0])
#Train["GarageFinish"] = Train["GarageFinish"].fillna(Train["GarageFinish"].value_counts().index[0])
#Train["GarageQual"] = Train["GarageQual"].fillna(Train["GarageQual"].value_counts().index[0])
#Train["GarageCond"] = Train["GarageCond"].fillna(Train["GarageCond"].value_counts().index[0])

#Train["GarageYrBlt"] = Train["GarageYrBlt"].fillna(Train["GarageYrBlt"].median())


# In[20]:


#print(Train["PoolQC"].unique())
#print(Train["Fence"].unique())
#print(Train["MiscFeature"].unique())

#Train["PoolQC"] = Train["PoolQC"].fillna(Train["PoolQC"].value_counts().index[0])
#Train["Fence"] = Train["Fence"].fillna(Train["Fence"].value_counts().index[0])
#Train["MiscFeature"] = Train["MiscFeature"].fillna(Train["MiscFeature"].value_counts().index[0])


# In[19]:


## for checking for null values after imputation
count = 0
for i in dataset.columns:   #### find out all the columns with null values
    print(i + ":", end = " ")
    print(dataset[i].isna().sum())
    if dataset[i].isna().sum() > 0:
        count +=1
print("total columns with null values : " + str(count))


# In[ ]:





# # Transformation of target variable 
# # only used in regression kind of problem

# In[20]:



get_ipython().run_line_magic('matplotlib', 'inline')
#Train['SalePrice'].hist(bins=50)
sns.distplot(Train['SalePrice'])


# In[21]:


#from scipy.stats import norm, skew
Train['SalePrice'] = np.log1p(Train['SalePrice'])
sns.distplot(Train['SalePrice']);


# # distinguinsh between categorical and numeric columns.

# In[62]:


## categorised for easy  preprocessing
numeric_data = dataset.select_dtypes(include=[np.number])
categorical_data = dataset.select_dtypes(exclude=[np.number])


# In[63]:


numeric_data


# In[26]:


#categorical_data
#categorical_cols = categorical_data.columns
#print(categorical_cols)


# In[24]:


from scipy.stats import skew
numeric_columns = dataset.select_dtypes(include=[np.number]).columns
skewed = dataset[numeric_columns].apply(lambda x: skew(x))
high_skew_data = skewed[abs(skewed) > 0.5]
high_skew_data


# In[26]:


for i in high_skew_data.index:
    dataset[i] = np.log1p(dataset[i])
    sns.distplot(dataset[i]);


# # Lable encoding and One hot Encoding to convert categorical values into numerical

# In[ ]:


### Lable encoding technique does not work here because it converts data into ordinal data

###trial = Train
### from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
####le = LabelEncoder()
# apply le on categorical feature columns
#### trial[categorical_cols] = trial[categorical_cols].apply(lambda col: le.fit_transform(col))


# In[27]:


# Using dummy variables
ohe_data = pd.get_dummies(categorical_data,drop_first=True)
ohe_data


# In[78]:


### using One hot encoding
#from sklearn.preprocessing import OneHotEncoder
#OHE = OneHotEncoder(drop='first')
#OHE.fit(categorical_data)
#OHE_data=OHE.transform(categorical_data).toarray()
#OHE_data_df=pd.DataFrame(OHE_data)
#OHE_data_df

from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder(drop='first').fit_transform(categorical_data).toarray()
OHE 


# In[79]:


OHE_df = pd.DataFrame(OHE)
OHE_df


# In[ ]:





# # merge two data frames

# In[59]:


#num_data= np.array(numeric_data)
#OHE_data = np.array(ohe_data)
#mixed_dataset = pd.concat([ohe_data, numeric_data], axis=1, sort=False)
#mixed_dataset


# In[86]:


##### Numeric columns remain first and then append categorical
#final_data = np.append(arr=num_data, values=OHE, axis=1)
##final_data = pd.DataFrame(final_data)
###final_data




#categorical columns remain first and then append numerical as our target variable is numeric variable.
#final_data = np.append(arr=OHE, values=num_data, axis=1)
#final_data = pd.DataFrame(final_data)
#final_data
Data


# # Feature Scaling
# 
# 

# In[ ]:


#train_features = final_data.drop([245],axis=1)
#train_target = final_data[245]
#train_features

#test_features = pd.read_csv(r"C:\Users\A\Desktop\kaggle\test.csv")
#test_features


# In[53]:


#from sklearn.preprocessing import RobustScaler
#scaler = RobustScaler() 
#scaled_data = scaler.fit_transform(mixed_dataset)
#scaled_data
#df= pd.DataFrame(scaled_data)
#df

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaled_data = scaler.fit_transform(mixed_dataset)
scaled_data
df= pd.DataFrame(scaled_data)
df


# In[33]:


#test_features_scaled_data= scaler.transform(test_features)
#test_features_scaled_data


# In[ ]:





# In[ ]:





# # Dimentionality Reduction using PCA

# In[81]:


#from sklearn.decomposition import PCA
#pca - keep 90% of variance
#pca = PCA(0.90)
#pca = PCA(n_components=2)
#principal_components = pca.fit_transform(df)
#principal_df = pd.DataFrame(data = principal_components)
#print(principal_df.shape)


# In[83]:


#principal_df


# # Feature selection 
# # this dataset consists categorical feature thats why we have to do it after one hot encoding otherwise before that

# In[ ]:


#import seaborn as sns
#import matplotlib.pyplot as plt
#%matplotlib inline
#Using Pearson Correlation
#plt.figure(figsize=(12,10))
#cor = final_data.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
#plt.show()


# In[ ]:


## def correlation(dataset, threshold):
   ### col_corr = set()  # Set of all the names of correlated columns
    ###corr_matrix = dataset.corr()
    ### for i in range(len(corr_matrix.columns)):
        ### for j in range(i):
            ### if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                ### colname = corr_matrix.columns[i]  # getting the name of column
                ### col_corr.add(colname)
    ### return col_corr


# In[ ]:


### corr_features = correlation(final_data, 0.9)
### set(corr_features)


# In[54]:


x_train =df[:len(y_train)]
x_test = df[len(y_train):]
print(x_train.shape)
print(x_test.shape)


# In[ ]:





# In[37]:


import xgboost as XGB

the_model = XGB.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1)
the_model.fit(x_train, y_train)


#from lightgbm import LGBMRegressor
#the_model = LGBMRegressor(n_estimators = 1000)
#the_model.fit(x_train,y_train)


# In[55]:



#y_predict = np.floor(np.expm1(the_model.predict(x_test)))
y_predict = the_model.predict(x_test)

#y_predict = the_model.predict(x_test)
y_predict


# In[39]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = y_predict
sub.to_csv(r'C:\Users\A\Desktop\kaggle\mysubmission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




