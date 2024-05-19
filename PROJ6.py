#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read data
df = pd.read_csv('Heart_Disease.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


print(f'the unique values of Gender: {df["Gender"].unique()}')
print(f'the unique values of work type: {df["work_type"].unique()}')
print(f'the unique values of smoking_status : {df["smoking_status"].unique()}')
print(f'the unique values of Heart Disease : {df["Heart Disease"].unique()}')
print(f'the unique values of Number of vessels fluro : {df["Number of vessels fluro"].unique()}')
print(f'the unique values of Slope of ST     : {df["Slope of ST"].unique()}')
print(f'the unique values of Exercise angina      : {df["Exercise angina"].unique()}')
print(f'the unique values of FBS over 120   : {df["FBS over 120"].unique()}')
print(f'the unique values of EKG results     : {df["EKG results"].unique()}')


# In[8]:


sns.countplot(x='Thallium', data=df, hue="Heart Disease", palette='BuPu')


# In[9]:


sns.countplot(x='Gender', data=df, hue="Heart Disease", palette='BuPu')


# In[10]:


##1-changed the object to numeric values
##2-dropped the id colum
##3-no duplicates
##4- removed the outliers
##n3ml eh tani??


# In[11]:


from sklearn import preprocessing 
label_encoder=preprocessing.LabelEncoder()
df['Gender']=label_encoder.fit_transform(df['Gender'])
df['work_type']=label_encoder.fit_transform(df['work_type'])
df['Heart Disease']=label_encoder.fit_transform(df['Heart Disease'])
df['smoking_status']=label_encoder.fit_transform(df['smoking_status'])
df.isnull().sum()


# In[12]:


print(df['Gender'].unique())  # null filled with 2
print(df['work_type'].unique()) # null filled with 5
print(df['Heart Disease'].unique()) 
print(df['smoking_status'].unique()) # null filled with 4   


# In[13]:


##df['Gender']=np.where(df['Gender']=='Female',0,1)
##df['work_type']= np.where(df['work_type']=='Private',0,np.where(df['work_type']=='Self-employed',1,np.where(df['work_type']=='Govt_job',2,np.where(df['work_type']=='children',3,4))))
##df['Heart Disease']=np.where(df['Heart Disease']=='No',0,1)
##df['smoking_status']= np.where(df['smoking_status']=='never smoked',0,np.where(df['smoking_status']=='smokes',1,np.where(df['smoking_status']=='formerly smoked',2,3)))
df.head(30)


# In[14]:


df['Gender']=df['Gender'].replace(2,np.nan)
df['work_type']=df['work_type'].replace(5,np.nan)
df['smoking_status']=df['smoking_status'].replace(4,np.nan)
df.head(30)


# In[15]:


mode_value = df['Gender'].mode().iloc[0]
df['Gender'] = df['Gender'].fillna(mode_value)

mode_value = df['work_type'].mode().iloc[0]
df['work_type'] = df['work_type'].fillna(mode_value)

mode_value = df['smoking_status'].mode().iloc[0]
df['smoking_status'] = df['smoking_status'].fillna(mode_value)


# In[16]:


df['Age'].fillna(value=df['Age'].mean(),inplace=True)
#df['Gender'].fillna(value=df['Gender'].mean(),inplace=True)
#df['work_type'].fillna(value=df['work_type'].mean(),inplace=True)
#df['smoking_status'].fillna(value=df['smoking_status'].mean(),inplace= True)


# In[17]:


df['Gender']=df['Gender'].astype(int)
df['work_type']=df['work_type'].astype(int)
df['smoking_status']=df['smoking_status'].astype(int)


# In[18]:


df.info()


# In[19]:


df.isnull().sum()


# In[20]:


df.head(20)


# In[21]:


sum(df.duplicated())


# In[22]:


df.drop(['id'], axis=1, inplace=True)
#df.drop(['Max HR'], axis=1, inplace=True)
#df.drop(['Gender'], axis=1, inplace=True)


# In[23]:


sns.boxplot(data=df[['Cholesterol']])


# In[24]:


sns.boxplot(data=df[['BP']])


# In[25]:


sns.boxplot(data=df[['ST depression']])


# In[26]:


columns = ['ST depression','BP','Cholesterol']
for col in columns:
    # calculate interquartile range
    q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = ( ( df[col] < lower) | (df[col] > upper) )
    index_label = df[outliers].index
    print(f'Number of outliers in {col}: {len(index_label)}')
    df.drop(index_label, inplace=True)


# In[27]:


sns.boxplot(data=df[['Cholesterol']])


# In[28]:


sns.boxplot(data=df[['BP']])


# In[29]:


sns.boxplot(data=df[['ST depression']])


# In[30]:


df.head()


# In[31]:


col=['Chest pain type','EKG results','Exercise angina','Slope of ST','Number of vessels fluro','work_type','smoking_status']


# In[32]:


for i in col[:7]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Heart Disease', data=df ,palette='plasma')
    plt.xlabel(i, fontsize=14)


# In[33]:


# plotting the correlation matrix
df.corr()
sns.set(rc={'figure.figsize': (15,10)})
sns.heatmap(df.corr(),annot=True)
#max HR is the most feature that affects the heart disease


# In[34]:


X = df.drop('Heart Disease',axis=1)
y = df['Heart Disease']


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# # Algorithms

# In[36]:


#LOGISTIC REGRESSION
classifier = LogisticRegression(random_state = 0)


# In[37]:


classifier.fit(X_train, y_train)


# In[38]:


y_pred = classifier.predict(X_test)


# In[39]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[40]:


cr=classification_report(y_test, y_pred)
alr=accuracy_score(y_pred,y_test)*100
print(cr)


# In[41]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[42]:


X_train=df.loc[:200].drop("Heart Disease",axis=1)
y_train=df.loc[:200]["Heart Disease"]

X_test=df.loc[201:].drop("Heart Disease",axis=1)
y_test=df.loc[201:]["Heart Disease"]


# In[43]:


model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)


# In[44]:


y_preds=model.predict(X_test)
KNN=accuracy_score(y_test,y_preds)*100


print(f"Testing score = {accuracy_score(y_test,y_preds)}")


# In[45]:


#DecisionTree model
DT = DecisionTreeClassifier(random_state=0)
DT.fit(X_train, y_train)


# In[46]:


y_predict = DT.predict(X_test)
#  prediction Summary by species
print(classification_report(y_test, y_predict))
# Accuracy score
DT_SC = accuracy_score(y_predict,y_test)*100
print(f"{round(DT_SC,2)}% Accurate")


# In[47]:


lasso_cv = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
y_PRed = lasso_cv.predict(X_test)
y_PRed = (y_PRed > 0.5).astype(int)


# In[48]:


lso = metrics.accuracy_score(y_test, y_PRed)*100
print('accuracy score:', lso)


# In[49]:


# Creating SVM model.
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_PRED = clf.predict(X_test)


# In[50]:


svm=accuracy_score(y_test,y_PRED)*100
print(accuracy_score(y_test,y_PRED))


# In[51]:


rf=RandomForestClassifier(random_state=6,n_estimators=100)
rf.fit(X_train,y_train)


# In[52]:


pred=rf.predict(X_test)
rfa=accuracy_score(y_test,pred)*100
print(rfa)


# In[53]:


score = [DT_SC,alr,KNN,lso,svm,rfa]
Models = pd.DataFrame({
'Algorithm': ["Decision Tree","Logistic Regression","KNN","lasso","SVM","rfa"],
'Score': score})
Models.sort_values(by='Score', ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




