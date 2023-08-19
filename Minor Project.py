#!/usr/bin/env python
# coding: utf-8

# # importing required libraries

# In[1]:


#Data handling
import pandas as pd
import numpy as np

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MinMaxScaler

#feature Scaling
from sklearn.feature_selection import mutual_info_classif

#Classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

#Perform Metrices
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score


# # Read Data

# In[2]:


#Reading data

df= pd.read_csv('cancer_gene_expression.csv')


# # Data Exploration & Cleaning 

# In[3]:


#Checking the number of smaples and features 
#NOTE: The last columns contains the labels, it is not considered as a feature

print(df.shape)


# In[4]:


#Checking some of the columns (first, second & third columns)

print(df.columns[0:3])


# In[5]:


#Checking the name of the last column of this dataframe

df.columns[-1]


# In[6]:


#Checking for any missing values

nullData= df.isnull().sum()
g= [i for i in nullData if i>0]

print('columns with missing values:%d'%len(g))


# In[7]:


#Checking how many different cancer types are there in th data
#NOTE: The cancer types will be referred to as classes or labels

print(df['Cancer_Type'].value_counts())


# In[8]:


#Plotting a bar chart to display the class distribution

df['Cancer_Type'].value_counts().plot.bar()


# # Data Preprocessing

# In[9]:


#We will now seperate the feature values from the class

x= df.iloc[:, 0:-1]
y= df.iloc[:, -1]


# In[10]:


print(x.shape)
print(y.shape)


# In[11]:


#Encoding target labels (y) with values between 0 and N_classes-1
#Here encoding will be done using labelEncoder

label_encoder= LabelEncoder()
label_encoder.fit(y)
y_encoded= label_encoder.transform(y)
labels= label_encoder.classes_
classes= np.unique(y_encoded)


# In[12]:


labels


# In[13]:


classes


# In[14]:


#Splitting of data into training and testing dataset

X_train, X_test, Y_train, Y_test= train_test_split(x,  y_encoded, test_size=0.2, random_state= 42)


# In[15]:


df.iloc[:,0:10].describe()


# In[16]:


#Data Normalization
#Scaling of data between 0 and 1

min_max_scaler= MinMaxScaler()
X_train_norm= min_max_scaler.fit_transform(X_train)
X_test_norm= min_max_scaler.fit_transform(X_test)


# In[17]:


type(X_train)


# In[18]:


X_train.iloc[0, 3]


# In[19]:


X_train_norm[0, 3]


# In[20]:


#Feature Seleciton using Mutual Information

MI= mutual_info_classif(X_train_norm, Y_train)


# In[21]:


MI.shape


# In[22]:


MI[0:5]


# In[23]:


features= X_train.columns


# In[24]:


features.shape


# In[25]:


features[0:5]


# In[26]:


#Selecting top n features, for example 300

n_features= 300
selected_scores_indices= np.argsort(MI)[::-1][0:n_features]


# In[27]:


X_train_selected= X_train_norm[:, selected_scores_indices]
X_test_selected= X_test_norm[:, selected_scores_indices]


# In[28]:


X_train_selected.shape


# In[29]:


X_test_selected.shape


# # Classification

# In[30]:


RF= OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
RF.fit(X_train_selected, Y_train)
Y_pred= RF.predict(X_test_selected)
pred_prob= RF.predict_proba(X_test_selected)


# In[33]:


#Accuracy
accuracy= np.round(balanced_accuracy_score(Y_test, Y_pred), 4)
print('Accuracy:%0.4f'%accuracy)

#Precision
precision= np.round(precision_score(Y_test, Y_pred, average= 'weighted'), 4)
print("Precision:%0.4f"%precision)

#Recall
recall= np.round(recall_score(Y_test, Y_pred, average='weighted'), 4)
print("Recall:%0.4f"%recall)

#f1score
f1score= np.round(f1_score(Y_test, Y_pred, average='weighted'), 4)
print('f1score:%0.4f'%f1score)

report= classification_report(Y_test, Y_pred, target_names=labels)
print('\n')
print("Classification report\n\n")
print(report)


# In[34]:


#Generating confusion matrix

cm= confusion_matrix(Y_test, Y_pred)
cm_df= pd.DataFrame(cm, index=labels, columns=labels)


# In[35]:


cm_df


# In[36]:


#Visualizing the confusion matrix using seaborn 

sns.heatmap(cm_df, annot=True, cmap='Blues')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")


# In[44]:


pred_prob.shape


# In[45]:


Y_pred.shape


# In[46]:




Y_test_binarized= label_binarize(Y_test, classes= classes)

fpr= {}
tpr= {}
thresh= {}
roc_auc= dict()

n_class= classes.shape[0]

for i in range(n_class):
    fpr[i], tpr[i], thresh[i]= roc_curve(Y_test_binarized[:,i], pred_prob[:,i])
    roc_auc[i]= auc(fpr[i], tpr[i])

#Plotting

plt.plot(fpr[i], tpr[i], linestyle='--',
        label= '%s vs Rest (AUC=%0.2f)'%(labels[i], roc_auc[i]))

plt.plot([0,1], [0,1],'b--')
plt.xlim([0,1])
plt.ylim([0, 1.05])
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

