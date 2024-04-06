#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing library: 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load the data
df = pd.read_excel('Sample_data.xlsx')
df


# In[3]:


# Renaming of columns Ticket_Summary for ISTM Ticket_Summary
df.rename(columns={'ITSM Ticket Summary': 'Ticket_Summary'}, inplace=True)
df


# ### Data Preprocessing

# In[4]:


# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[5]:


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# In[6]:


# Apply preprocessing to Ticket Summary data
df['Ticket_Summary'] = df['Ticket_Summary'].apply(preprocess_text)


# In[7]:


# Vectorize text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Ticket_Summary'])


# In[8]:


# Perform KMeans clustering
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
df['Cluster Label'] = kmeans.labels_


# In[9]:


df


# In[10]:


# Define cluster labels
cluster_labels = {
    0: "Database Management",
    1: "Job Execution",
    2: "File Management",
    3: "System Access",
    4: "Application Monitoring"
}


# In[11]:


df['Cluster Label'] = df['Cluster Label'].map(cluster_labels)
# Save the DataFrame to a CSV file
df.to_csv('modified_dataframe.csv', index=False)





# In[12]:


# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X, df['Cluster Label'], test_size=0.2, random_state=42)


# In[13]:


# Dictionary to store results
results = {}


# In[14]:


# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(kernel='linear'),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}


# In[15]:


# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy


# In[16]:


# Print results
print("Accuracy for each model:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy}")


# In[21]:


# Create an instance of the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict using the trained classifier
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))


# In[22]:


rf_classifier


# In[ ]:





# In[23]:


# Save the model as a pickle file
with open('rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# After fitting the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# In[ ]:




