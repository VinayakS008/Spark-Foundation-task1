#!/usr/bin/env python
# coding: utf-8

# # The spark Foundation : Data Science and business analytics intership
# 
# Task 1 : Prediction using Supervised Machine Learning Problem
# 
# Statement : predict the percentage of an student based on the no.of study hours
# 
# Author : Sapkal Vinayak Dadaso
# 

# In this task 'Simple Linear Regression ' will be performed on 'Scores of students and their study hours', the dataset contain two variables study hours and their scores.

# # Step 1 : Processing the Dataset

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Reading the dataset
data=pd.read_csv("C:/Users/Vinayak/Desktop/student_scores - student_scores.csv")
data.head()


# In[4]:


#Rows and columns in dataset
data.shape


# In[5]:


#Information about data
data.info()


# In[6]:


#Descripltive statistics
data.describe()


# Mean average of study hours is 5 hours, minimum study hours is 1 hours and maximimum study hours are 9 hours
# mean average of scores is 51 %,minimum score is 17% and maximium score is 95% 

# In[7]:


# To check missing values in data
data.isnull().sum()


# There are no missing values in dataset so we move forward to our next step

# # Step 2 : Visualization of data

# To check relationship between the two variable linear regression very effective used to predict the scores on the number of hours.

# In[8]:


import matplotlib.pyplot as plt


# In[9]:


plt.scatter(x=data.Hours, y=data.Scores)
plt.title('Hours Vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# > The above Scatter Plot Shows the Relationship Between Students Study Hours and Their respective scores.
# > From above graph we conclude that as study hours increasing then marks also increasing
# 

# In[10]:


ax = sns.heatmap(data.corr(),annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.4,top-0.4)
plt.show()


# From above heatmap it is observed that their is positive correlation between hours studied and scores of students.

# # Step 3 : Data Preperation

# In[11]:


# Dividing the data in training and testing data
data.head()


# In[12]:


# Dividing the dataset
X = data.iloc[:,:1].values
y = data.iloc[:,1:].values


# In[16]:


X


# In[14]:


y


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Step 4 : Training the model 

# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


model=LinearRegression()
model.fit(X_train,y_train)


# # Step 5 : Visualize the model

# In[21]:


# Plotting the testing dataset
plt.rcParams["figure.figsize"]=[14,8]
plt.scatter(X_test, y_test , color='red')
plt.plot(X,line, color = 'green')
plt.title('Hours vs Score')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# From above graph the line give us quit good prediction about testing dataset

# In[20]:


# Plotting the training dataset
line = model.coef_*X + model.intercept_

plt.rcParams["figure.figsize"]=[14,8]
plt.scatter(X_train, y_train , color='red')
plt.plot(X,line, color = 'blue')
plt.title('Hours vs Score')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[ ]:


From above graph the line give us good prediction about traning dataset


# # Step 6 : Prediction for training and testing dataset

# In[22]:


print(X_test)
y_pred = model.predict(X_test)


# In[23]:


y_test


# In[24]:


y_pred


# In[25]:


# To compare actual and predicted value 
comp = pd.DataFrame({'Actual':[y_test],'Predicted':[y_pred]})
comp


# In[26]:


# Testing with original data
hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score for if a student studied for ",hours,"hours is", own_pred[0])


# Therefore, if a student studied for 9.25 hours then the predicted score of student is 93.6917

# In[ ]:




