
# coding: utf-8

# # Introduction
# 
# The data available in this problem contains the following information, including the details of a sample of campaigns and coupons used in previous campaigns 
# 
# - User Demographic Details
# - Campaign and coupon Details
# - Product details
# - Previous transactions
# 
# Data is available for previous 18 Campaigns for coupon and customer combination
# 
# Following Files are available for analysis,
# - Test Data (csv)
# - Train Data (csv)
# - Submissions(csv)

# # Problem Statement
# 
# Predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?
# 

# # Hypothesis Generation
# 
# Below are some factors that may influence Redemption Status
# 
# - Campaign : Customers can redeem the given coupon for any valid product for that coupon as per coupon item mapping within the duration between campaign start date and end date
# - Coupons : Customers receive coupons under various campaigns and may choose to redeem it

# # Loding the necessary data

# In[155]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# # Reading the Data

# In[156]:


train = pd.read_csv("D:/hack/AM/train.csv")
test = pd.read_csv("D:/hack/AM/test.csv")
train_original = train.copy()
test_original = test.copy()


# # Structure of the Data Set

# In[157]:


train.columns


# In[158]:


test.columns


# In[159]:


train.dtypes


# In[160]:


test.dtypes


# In[161]:


train.shape


# In[162]:


train.info


# In[163]:


test.shape


# # Key Observations - Structure of Data
# 
# * Train Dataset contains 78369 rows and 5 columns
# * Test Data contains 50226 rows and 4 columns
# * Train Data contains Features with datatype numerical interger
# * Test Data contains Features with datatype numerical integer

# #  Univarient Analysis of the Dataset

# In[164]:


train['redemption_status'].value_counts()


# In[165]:


train['redemption_status'].value_counts(normalize = True)


# In[166]:


train['redemption_status'].value_counts().plot.bar(title = 'Redemption_Status')


# # Observations - Target variable
# 
# - Out of 78369 , 729 (0.93%) of customers have redemmed the coupon

# # Independent Variables (Univarient Analysis)
# 
# -Campaign id
# -Coupon id
# -Customer id

# In[167]:


# Campaign ID
sns.distplot(train['campaign_id'])


# In[168]:


train['campaign_id'].plot.box(figsize=(16,5))


# # Inference - Univarient Analysis ( campaign id)
# 
#  - It is inferred that data in the distribution of campaign id is towards left ,the same is not normall distributed.
#  - We will make the same normal to train the algorithm better
#  - The Boxplot confirms very feabile presence of outliers/extreme values
#  

# In[169]:


# Coupon id
sns.distplot(train['coupon_id'])


# In[170]:


train['coupon_id'].plot.box(figsize=(16,5))


# # Inference- Univarient Analysis (Coupon_id)
# 
# - It is inferred that data from couponid id fairly normally distributed
# - The boxplot confirms the absence of extreme outliers points for the same

# In[171]:


# Customer_id
sns.distplot(train['customer_id'])


# In[172]:


train['customer_id'].plot.box(figsize=(16,5))


# # Inference - Univarient Analysis (Customer_id)
# 
# - It is inferred that data in distribution of customer_id is fairly normally distriuted
# - Absence of outliers concurs the same

# # Bivarient Analysis -  To analyse correlation between independent and Target Features

# In[173]:


# Bivarient analysis between Campaign vs Redemption_Status
train.groupby('redemption_status')['campaign_id'].mean().plot.bar(title = "Campaign Vs Redemption_Status")


#  Here the y-axis represents the discount campaign. We don’t see any change in the mean 

# In[174]:


# Bivarient analysis between discount Coupon vs Redemption_Status
train.groupby('redemption_status')['coupon_id'].mean().plot.bar(title = "Coupon Vs Redemption_Status")


#  Here y axis indicates mean discount coupon , we see increase in redemption status against discount coupons

# In[175]:


# Bi varient analysis between Customer and redemption_status
train.groupby('redemption_status')['customer_id'].mean().plot.bar(title = "Coupon Vs Redemption_Status")


# Customers who redem coupons are more from the above graph

# # Multi- Colleniarity & Correlation analysis between Features

# In[176]:


matrix = train.corr()


# In[177]:


sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[178]:


train.corr()


# # Inference  - Correlation Analysis
# 
# - Values close to +1 - denotes strong positive correlation
# - Values close to -1 - denoates strong negative correlation
# -  Above we donot interpret both positive or negative correlation

# # Missing (?) values and Outlier Analysis

# In[179]:


train.isnull().sum()


# In[180]:


test.isnull().sum()


# # Inference - Missing(?) values
# - No missing values were observed on both train and test datasets

# In[181]:


train.hist()


# Data is Fairly normally distributed

# In[182]:


train.info
train.head(10)


# In[183]:


test=test.drop('id',axis=1)
train = train.drop('id',axis =1)


# # spliting variables into Indepedent and Dependent

# In[184]:


X = train.drop('redemption_status',1)
y = train.redemption_status


# # Train_Test_Split

# In[185]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.4)


# In[186]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


# # Training Logistic Algorithm , analyzing accuracy

# In[187]:


model = LogisticRegression() 
model.fit(x_train, y_train)


# In[188]:


pred_cv = model.predict(x_cv)


# In[189]:


accuracy_score(y_cv,pred_cv)


# In[190]:


pred_test_Logit = model.predict(test)


# # Random Forest Algorithm

# In[192]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state = 1 ,n_estimators = 20)
rf_model.fit(x_train,y_train)


# In[194]:


pred_cv_rf = rf_model.predict(x_cv)


# In[195]:


accuracy_score(y_cv,pred_cv_rf)


# In[196]:


pred_test_rf = rf_model.predict(test)


# # Decision - Tree

# In[197]:


from sklearn import tree
dt_model = tree.DecisionTreeClassifier(random_state = 1)
dt_model.fit(x_train,y_train)


# In[198]:


pred_cv_dt = dt_model.predict(x_cv)


# In[199]:


accuracy_score(y_cv,pred_cv_dt)


# In[200]:


pred_test_dt = dt_model.predict(test)


# # Observations 
# 
# - Logistic regression seems to project a good accuracy score when compared to rest of the models , let us proceed to use the model predictions to predict the balance 10 redemption status in test set against coupon and customer combination

# In[203]:


importances = pd.Series(rf_model.feature_importances_, index = X.columns)
importances.plot(kind = 'barh', figsize =(12,8))


# In[204]:


submission = pd.read_csv('D:/hack/AM/submission.csv')


# In[207]:


submission['redemption_status'] = pred_test_Logit


# In[208]:


submission['id'] = test_original['id']


# In[209]:


submission['redemption_status'].replace(0,'N',inplace = True)
submission['redemption_status'].replace(1,'Y',inplace =True)


# # Retreiving the output file in local path after model prediction

# In[210]:


pd.DataFrame(submission, columns=['id','redemption_status']).to_csv('D:/hack/AM/Redemption.csv')

