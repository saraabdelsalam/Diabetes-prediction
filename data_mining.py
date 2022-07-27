#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("Dataset of Diabetes .csv")
df.head()
#df.info()


# In[2]:


df.isnull().sum()

df.duplicated().sum()

le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
df['Gender']
df["CLASS"]=le.fit_transform(df["CLASS"])


df.drop("ID",axis=1,inplace=True) 
df.drop("No_Pation",axis=1, inplace=True) 
df


x=df.iloc[:,0:11]
y=df["CLASS"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

x_train


# In[3]:


#----------------------Decision_Tree---------------------------------------
classifier1=DecisionTreeClassifier()
#accuracy=cross_val_score(classifier1 ,x_train,y_train , scoring='accuracy'  , n_jobs= -1 ).mean()
#print(accuracy)
classifier1.fit(x_train,y_train)
predection= classifier1.predict(x_test)
accuracy=accuracy_score(y_test,predection)
names=list(df.columns.values)
plt.figure(figsize=(8,8))
figure=plot_tree(classifier1,feature_names=names,max_depth=4,fontsize=10,label='root',class_names=True)
print("-------------------DECISION TREE ACCURACY----------------------------")
print("*DECISION TREE ACCURACY* \n")
print("Final Accuracy is :\n ",accuracy)


# In[4]:


#-----------------------Naive_bayes-----------------------------------------
GA = GaussianNB()
GA.fit(x_train, y_train)
y_pred = GA.predict(x_test)
#print(y_pred)
print("---------------------GUASSIAN NB ACCURACY-------------------------------")
print("*GUASSIAN NB ACCURACY*\n")
print("Final Accuracy is :\n", metrics.accuracy_score(y_test, y_pred))
#F_score=f1_score(y_test,y_pred,average='macro')
#print("F1_score :\n",F_score)


# In[5]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#print(y_pred)
print("------------------------K-NEIGHBORS ACCURACY--------------------------")
print("*K-NEIGHBORS ACCURACY*\n")
print("Final Accuracy is:\n",metrics.accuracy_score(y_test, y_pred))
#F_score=f1_score(y_test,y_pred,average='macro')
#print("\n")
#print("F1_score:\n",F_score)


# In[7]:


#-----------------------------SVM--------------------------------------------

classifier= SVC(kernel='linear',random_state=0)
classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
print("------------------------SVC ACCURACY----------------------------------")
print("*SVC ACCURACY*\n")
print("Final Accuracy is :\n", metrics.accuracy_score(y_test, y_predict))


# In[8]:


#-------------------------Final Result----------------------------------
print("-------------------------FINAL ANSWERS----------------------------------")
print (" 1) Percentage of younger people that are prone to diabetes is : (4/35 = 0.075)\n", "Final percentage is 0.075%\n")
print (" 2) Percentage of women that are prone to diabetes is : (17/35 = 0.321)\n", "Final percentage is 0.321%\n")
prinb=("So women is less prone to diabetes than males")

"""diabetic = 844
non-diabetic = 103
predicate = 53
"""
df["Cr"].value_counts()
df["HbA1c"].value_counts()
df["CLASS"].value_counts()
df.loc[df["BMI"] >=30, "CLASS"].value_counts()
df["CLASS"].describe()
df.loc[df["BMI"]<30].value_counts()


# In[9]:


# visualization_countplot
sns.countplot(x = "CLASS",data = df)


# In[10]:


#visualization_each feature
for i in df.iloc[:,1:11]:    
    plt.figure()
    plt.title(f'{i}')
    plt.hist(df[i])
    


# In[11]:


#heatmap_visualization
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[13]:


#kmeams
#df["CLASS"]=le.fit_transform(df["CLASS"])
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(df)
plt.scatter(df["CLASS"],df["Gender"])
clusters=kmeans.cluster_centers_
KM=kmeans.fit_predict(df[["CLASS","Gender"]])
print(KM)
plt.scatter(df.CLASS,df["Urea"],s=50,color='blue')
plt.scatter(df.CLASS,df["BMI"],s=50,color='black')
plt.scatter(df.CLASS,df["AGE"],s=50,color='cyan')
plt.scatter(df.CLASS,df["Gender"],s=50,color='purple')
plt.show()


# In[ ]:




