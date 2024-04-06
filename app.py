from flask import Flask, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
with open('rf_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load your dataset containing ticket information
# Let's assume your dataset is in a CSV file named "ticket_data.csv"
ticket_data = pd.read_csv('modified_dataframe.csv')

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def average_resolution_time(cluster_label):
    # Filter the dataset for tickets belonging to the given cluster label
    cluster_tickets = ticket_data[ticket_data['Cluster Label'] == cluster_label]
    
    # Convert the date columns to datetime objects
    cluster_tickets['ITSM Ticket Resolved Date'] = pd.to_datetime(cluster_tickets['ITSM Ticket Resolved Date'])
    cluster_tickets['ITSM Ticket Opened Date'] = pd.to_datetime(cluster_tickets['ITSM Ticket Opened Date'])
    cluster_tickets['ITSM Ticket End Date'] = pd.to_datetime(cluster_tickets['ITSM Ticket End Date'])
    
    # Calculate the resolution time for each ticket
    def calculate_resolution_time(row):
        if pd.notnull(row['ITSM Ticket Resolved Date']):
            return row['ITSM Ticket Resolved Date'] - row['ITSM Ticket Opened Date']
        elif pd.notnull(row['ITSM Ticket End Date']):
            return row['ITSM Ticket End Date'] - row['ITSM Ticket Opened Date']
        else:
            return pd.NaT  # Return NaT for tickets with missing resolved and end dates

    cluster_tickets['Resolution Time'] = cluster_tickets.apply(calculate_resolution_time, axis=1)

    # Drop tickets with missing resolution times
    cluster_tickets = cluster_tickets.dropna(subset=['Resolution Time'])

    if not cluster_tickets.empty:
        # Calculate the average resolution time for the cluster
        avg_resolution_time = cluster_tickets['Resolution Time'].mean()
        
        # Convert average resolution time to a string representation (you may want to format it properly)
        avg_resolution_time_str = str(avg_resolution_time)
    else:
        avg_resolution_time_str = None  # Handle case where there are no tickets with resolved or end dates
    
    return avg_resolution_time_str

# Endpoint to predict cluster label and average resolution time
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticket_summary = data['Ticket_Summary']
    
    # Preprocess text
    preprocessed_summary = preprocess_text(ticket_summary)
    
    # Vectorize preprocessed text
    vectorized_summary = vectorizer.transform([preprocessed_summary])
    
    # Predict cluster label
    predicted_label = model.predict(vectorized_summary)[0]
    
    # Calculate average resolution time for the predicted cluster
    avg_resolution_time = average_resolution_time(predicted_label)
    
    # Prepare response
    response = {
        "Predicted_Cluster_Label": predicted_label,
        "Average_Resolution_Time": avg_resolution_time
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
