#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from warnings import filterwarnings
filterwarnings("ignore")
plt.rcParams["figure.figsize"]=[15,10]

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import shapiro
import statsmodels.stats.api as ssa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sma
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


# In[2]:


df = pd.read_excel("data.xlsx")
df.head()


# In[3]:


df.info()


# In[4]:


df.corr()


# In[5]:


df.describe()


# In[6]:


cols = ['water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)']


# In[7]:


for i in cols:
    print(df[i].value_counts())


# In[8]:


num_cols = df.select_dtypes(exclude='object')
num = num_cols.columns

nrows= 2
ncol = 2
iterator = 1
for i in num:
    plt.subplot(nrows,ncol,iterator)
    sns.distplot(df.loc[:,i],kde=True)
    iterator +=1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[9]:


plt.rcParams["figure.figsize"]=[8,12]
cols = ["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)']
nrows= 3
ncol = 1
iterator = 1
for i in cols:
    plt.subplot(nrows,ncol,iterator)
    sns.violinplot(x=df[i],y=df['Ra(output)'])
    iterator+=1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[10]:


plt.rcParams["figure.figsize"]=[10,5]
sns.heatmap(df.corr(),annot=True,cmap='coolwarm', fmt=".2f")
plt.show()


# In[11]:


sns.pairplot(df,palette="bright")
plt.show()


# In[12]:


df1 = df.copy(deep=True)


# #### Feature engineering

# In[13]:


# because the water pressure has the highest coorelation with the target


# In[14]:


df1['Mean_Targ_Traverse_rate'] = df1.groupby('Traverse rate (mm/min)')['Ra(output)'].transform('mean')


# #### Scaling the data
# * using the standard scaler transformation

# In[15]:


ss = StandardScaler()


# In[16]:


df1[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)']] = ss.fit_transform(df1[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)']])


# **Statistical test** 

# * Ho: that the predictor is independent of target ( the predictor and target are not related)
# * H1: that the predictor dependent of target

# In[17]:


cols


# In[18]:


for i in cols:    
    teststats,pvalue = stats.ttest_ind(df1[i], df1['Ra(output)'])
    print(i,'=',pvalue,'/',(pvalue<0.05))


# p value is < 0.05 for all the independent varialbles, 
# hence we conclude that the target variable is dependent on all the features

# In[19]:


df1.skew()


# In[20]:


# there is no skewness in the data


# In[21]:


# model builiding


# In[22]:


x = df1.drop(columns=['Ra(output)'],axis=1)
y = df1['Ra(output)']


# In[23]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.8,random_state=10)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[24]:


temp_train = sma.add_constant(xtrain)
temp_test = sma.add_constant(xtest)


# In[25]:


model = sma.OLS(ytrain,temp_train).fit()
model.summary()


# **Model Assumptions**
# * **Linearity** : That there should be a linear pattern between the predictors and target ** Statistical test : rainbow Test**
# * **Normality** : That the model residuals should be normal ** Statisticak Test : Jarque bera test**
# * **Multicollinearity** : All the predictors should not be highly correlated, ** Statistical test : Correlation,condition NO & VIF**
# * **Autocorrelation of Errors**: the error should not be correlated ** Statistical Test: Durbin Watson Test the Range of DW test is between 0-4 where is the ideal value is 2 and tolerance range 1.5-2.5
# * **Heteroscedasticity** : That the data has unequal variance , Statistical Test : Breusch Pagan Test

# In[26]:


# Linearity : That there should be a linear pattern between the predictors and target --> Statistical test : rainbow Test
ssa.linear_rainbow(model)[1]


# In[27]:


# pvalue > 0.05 --> Linearity is passed


# In[28]:


ssa.jarque_bera(model.resid)[1] #  pvalue > 0.05 --> Linearity is passed


# In[29]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[30]:


# Multicollinearity  - condition no is below 100 then its ok
vif_list=[]
for i in range(xtrain.shape[1]):
    vif_list.append(variance_inflation_factor(xtrain.values,i))

pd.DataFrame({'features':xtrain.columns,'VIF':vif_list}).sort_values(by='VIF',ascending=False)


# In[31]:


# Autocorrelation of Errors

from statsmodels.stats.stattools import durbin_watson
durbin_watson(model.resid)
# 1.5-2.5 ( no auto correlation ) 


# In[32]:


# Heteroscedasticity
ssa.het_breuschpagan(model.resid,model.model.exog)[3] 
# That the data has unequal variance


# In[33]:


# all the model assumptions  are passed 


# In[ ]:





# In[34]:


# Creating a linier model for data size increase


# In[ ]:





# In[35]:


df.head()


# In[36]:


x_up = df.drop(columns=['Ra(output)'],axis=1)
y_up = df['Ra(output)']


# In[37]:


lr = LinearRegression() 

lr.fit(x_up,y_up)


# In[38]:


lr.coef_


# In[39]:


lr.intercept_


# In[40]:


df.describe()


# ### Increasing sample size

# In[ ]:





# In[41]:


def create_dataset():
    names = ['water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)','Ra(output)']
    df = pd.DataFrame([], columns=names)
    total = 0
    while total < 300:
        total += 1 
        
        import random
        num1 = 0    
        num1 = random.randint(193,234)
        num2 = random.randint(3,5)    
        num3 = random.randrange(15,36,5)

        Ra1 = 0.00165028
        Ra2 = 0.01416667
        Ra3 = -0.00087778
        Ra_in = 1.2566630099568248

        CIR = 0
        CIR = num1*Ra1
        CIR = CIR + num2*Ra2
        CIR = CIR + num3*Ra3
        CIR = CIR + Ra_in
        CIR = round(CIR, 6)

        I1 = num1
        I2 = num2
        I3 = num3
        
        Target1 = CIR
                       
        vector = [I1,I2,I3,Target1]
        
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        
    return df


# In[42]:


df2 = create_dataset()


# In[43]:


df2.shape  # sample 


# In[44]:


df2.reset_index(inplace=True)


# In[45]:


df2.drop(columns='index',axis=1,inplace=True)


# In[46]:


df2.head()


# In[47]:


df2['water pressure (Mpa)'] = df2['water pressure (Mpa)'].astype('int')
df2['SOD(mm)'] = df2['SOD(mm)'].astype('int')
df2['Traverse rate (mm/min)'] = df2['Traverse rate (mm/min)'].astype('int')


# In[48]:


df2.info()


# In[49]:


df2.corr()


# In[50]:


df2.describe()


# In[51]:


num_cols = df2.select_dtypes(exclude='object')
num = num_cols.columns

nrows= 2
ncol = 3
iterator = 1
for i in num:
    plt.subplot(nrows,ncol,iterator)
    sns.distplot(df2.loc[:,i],kde=True)
    iterator +=1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[52]:


plt.rcParams["figure.figsize"]=[15,15]
cols = ['SOD(mm)', 'Traverse rate (mm/min)']
nrows= 4
ncol = 1
iterator = 1
for i in cols:
    plt.subplot(nrows,ncol,iterator)
    sns.violinplot(x=df2[i],y=df2['Ra(output)'])
    iterator+=1
    plt.title(i)
plt.tight_layout()
plt.show()


# In[53]:


plt.rcParams["figure.figsize"]=[8,5]
sns.heatmap(df2.corr(),annot=True,cmap='coolwarm', fmt=".2f")
plt.show()


# In[54]:


sns.pairplot(df2,palette="bright")
plt.show()


# In[55]:


df3 = df2.copy(deep=True)


# In[56]:


# feature engineering
df3['Mean_Targ_SOD(mm)'] = df3.groupby('SOD(mm)')['Ra(output)'].transform('mean')


# In[57]:


df3.skew()


# In[58]:


# from sklearn.preprocessing import PowerTransformer


# In[59]:


# a1 = ['Vol %', 'water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)', 'Mean_Targ_Vol%','Mean_Targ_SOD(mm)']


# In[60]:


# b=PowerTransformer(method='yeo-johnson')

# df3[a1] = b.fit_transform(df3[a1])


# In[61]:


df3.skew()


# In[ ]:





# In[62]:


df3.columns


# In[63]:


cols_1=[ 'water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)', 'Mean_Targ_SOD(mm)']


# In[64]:


df3[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)','Mean_Targ_SOD(mm)']] = ss.fit_transform(df3[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)','Mean_Targ_SOD(mm)']])


# In[65]:


x = df3.drop(columns=['Ra(output)'],axis=1)
y = df3['Ra(output)']


# In[66]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.8,random_state=10)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# In[67]:


from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.feature_selection import RFE


# In[68]:


# not using linear regression as data is created using linear regression
# the assumptions alos fails


# In[69]:


temp_train = sma.add_constant(xtrain)
temp_test = sma.add_constant(xtest)


# In[70]:


model1 = sma.OLS(ytrain,temp_train).fit()


# In[71]:


model1.summary()


# In[72]:


# Linearity : That there should be a linear pattern between the predictors and target --> Statistical test : rainbow Test
ssa.linear_rainbow(model1)[1]


# In[73]:


ssa.jarque_bera(model1.resid)[3]


# In[74]:


durbin_watson(model1.resid)


# In[75]:


ssa.het_breuschpagan(model1.resid,model1.model.exog)[3]   # 


# In[76]:


# so we reject all the parametric models for model bilding because the data is upsampled using linear regression 


# In[ ]:





# In[77]:


df5 = df2.copy(deep=True)


# In[78]:


df5[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)']] = ss.fit_transform(df5[["water pressure (Mpa)",'SOD(mm)', 'Traverse rate (mm/min)']])
x = df5.drop(columns=['Ra(output)'],axis=1)
y = df5['Ra(output)']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.8,random_state=10)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# ### Buliding Models

# In[79]:


dt   = DecisionTreeRegressor(random_state=10)
rf   = RandomForestRegressor(random_state=10)
knn  = KNeighborsRegressor()
Ada  = AdaBoostRegressor(random_state=10)
xgb  = XGBRegressor(random_state=10)


# In[80]:


l1 = [dt,rf,knn,Ada,xgb]


# In[81]:


def models(mod, xtrain, ytrain, xtest, ytest):
    l = [str(mod).split('(')[0]]
    model1 = mod.fit(xtrain, ytrain)
    ypred_train = model1.predict(xtrain)
    ypred = model1.predict(xtest)
    
    l.append(model1.score(xtrain, ytrain))  # Training accuracy
    l.append(model1.score(xtest, ytest))    # Testing accuracy
    
    l.append(round(r2_score(ytest, ypred),3))
    l.append(round(np.sqrt(mean_squared_error(ytrain, ypred_train)), 5))
    l.append(round(np.sqrt(mean_squared_error(ytest, ypred)), 5))  # rmse test
    
    return l


# In[82]:


best = pd.DataFrame(columns = ['Model','Train_ACC','Test_ACC','r2_score','RMSE_train','RMSE_test'])


# In[83]:


perf1 = models(dt,xtrain,ytrain,xtest,ytest)
perf2 = models(rf,xtrain,ytrain,xtest,ytest)
perf3 = models(knn,xtrain,ytrain,xtest,ytest)
perf4 = models(Ada,xtrain,ytrain,xtest,ytest)
perf5 = models(xgb,xtrain,ytrain,xtest,ytest)


# In[84]:


best.loc[len(best)] = perf1
best.loc[len(best)] = perf2
best.loc[len(best)] = perf3
best.loc[len(best)] = perf4
best.loc[len(best)] = perf5


# In[85]:


best


# In[86]:


# Ada boost is best model 
# doing hyper parameter tuning 


# In[87]:


AdaBoostRegressor()


# In[88]:


# hyper
# params = {'n_estimators':[100,150,200],
#          'learning_rate':[1.0,0.8,1.5,2]}

# Grid  = GridSearchCV(estimator=Ada,cv=5,param_grid=params,n_jobs=-1)

# Grid.fit(xtrain,ytrain)

# Grid.best_params_


# In[89]:


ada   = AdaBoostRegressor(n_estimators=150,random_state=10,learning_rate=1)
m1 = ada.fit(xtrain, ytrain)
ypred_train = m1.predict(xtrain)
ypred = m1.predict(xtest)

print(m1.score(xtrain, ytrain))  # Training accuracy
print(m1.score(xtest, ytest))    # Testing aprint
print(round(r2_score(ytest, ypred),3))
print(round(np.sqrt(mean_squared_error(ytrain, ypred_train)), 5))
print(round(np.sqrt(mean_squared_error(ytest, ypred)), 5))  # rmse test


# In[90]:


min(ypred)


# In[91]:


feature_importances = m1.feature_importances_


# In[92]:


feature_importances


# In[93]:


column_names = [ 'water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)']

importance_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importances})

importance_df.sort_values(by='Importance', ascending=False, inplace=True)

importance_df


# In[94]:


# Assuming you have the feature importances and the original DataFrame
column_names = ['water pressure (Mpa)', 'SOD(mm)', 'Traverse rate (mm/min)']

# Create a DataFrame to store the feature importances along with their corresponding column names
importance_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Get the best value for each feature
best_values = {}

for feature_name in importance_df['Feature']:
    best_values[feature_name] = df5[feature_name].max()

print("Best values for each feature:")
for feature, value in best_values.items():
    print(f"{feature}: {value}")


# In[95]:


df5[(df5['water pressure (Mpa)']==1.7231438996283654)|(df5['SOD(mm)']==1.2655243193355703)&(df5['Traverse rate (mm/min)']==1.767862923981746) ]


# In[96]:


df2.iloc[[15,27,49,53,77]]


# In[97]:


def preprocess(df2):
    X=df2.drop('Ra(output)',1)
    y=df2['Ra(output)']
    xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,random_state=42)
    return xtrain,xtest,ytrain,ytest


# In[98]:


# Creatig a pipline object for deployment
from sklearn.pipeline import Pipeline

steps = [('scaler',ss),('RFC tuned',Ada)]
pipeline = Pipeline(steps)
xtrain,xtest,ytrain,ytest = preprocess(df2)
pipeline.fit(xtrain,ytrain)
ypred = pipeline.predict(xtest)

print(pipeline.score(xtrain, ytrain))  # Training accuracy
print(pipeline.score(xtest, ytest))    # Testing aprint
print(round(r2_score(ytest, ypred),3))
#print(round(np.sqrt(mean_squared_error(ytrain, ypred_train)), 5))
print(round(np.sqrt(mean_squared_error(ytest, ypred)), 5))  # rmse test


# In[99]:


import pickle
model=open('rf.pickle','wb')
pickle.dump(pipeline,model)
model.close()


# In[100]:


get_ipython().run_cell_magic('writefile', 'app.py', "import streamlit as st\nimport pandas as pd\nimport numpy as np\nimport pickle\n\nst.title('Water_ Jet')\n\n# Step 1: Load the Model\nmodel = open('rf.pickle', 'rb')\npipeline = pickle.load(model)\nmodel.close()\n\n# Step 2: Create a UI for the front-end user\nwater_pressure = st.number_input('water pressure (Mpa)', min_value=193, max_value=234)\nSOD = st.number_input('SOD(mm)', min_value=3, max_value=7)\nTraverse_rate = st.number_input('Traverse rate (mm/min)', min_value=15, max_value=35)\n\n# Step 3: Change User Input to Model Input data\ndata = {'water_pressure': [water_pressure], 'SOD': [SOD], 'Traverse_rate': [Traverse_rate]}\ninput_data = pd.DataFrame(data)\n\n# Step 4: Get Predictions and Print the result\nif st.button('Predict'):\n    result = pipeline.predict(input_data)\n    st.table(input_data)\n    st.success(str(result[0]))")


# In[101]:


# streamlit run app.py


# In[ ]:


# streamlit run your_script.py


# In[ ]:




