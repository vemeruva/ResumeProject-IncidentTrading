import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import os

# Generate synthetic data
def generate_synthetic_data(num_records=1000, file_path="synthetic_incidents.csv"):
    """Generate synthetic incident data and save to CSV."""
    categories = ['Hardware', 'Software', 'Network', 'Security', 'User Support']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    descriptions = [
        "Server crashed during peak hours",
        "Application failed to load",
        "Network connectivity issue in office",
        "Unauthorized access detected",
        "User unable to reset password",
        "Printer not responding",
        "Software update caused errors",
        "Firewall configuration issue",
        "Slow system performance",
        "Email service outage"
    ]
    
    data = {
        'ticket_id': [f'INC{i:05d}' for i in range(1, num_records + 1)],
        'description': [random.choice(descriptions) + f" (Details: {random.randint(100, 999)})" for _ in range(num_records)],
        'category': [random.choice(categories) for _ in range(num_records)],
        'date_reported': [datetime(2025, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(num_records)],
        'resolution_time': [round(random.uniform(0.5, 72.0), 2) for _ in range(num_records)],
        'priority': [random.choice(priorities) for _ in range(num_records)]
    }
    
    df = pd.DataFrame(data)
    df['date_reported'] = pd.to_datetime(df['date_reported'])
    df.to_csv(file_path, index=False, date_format='%Y-%m-%d')
    return df, file_path

# Load and preprocess data
def load_data(file_path):
    """Load incident ticket data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['ticket_id', 'description', 'category', 'date_reported', 'resolution_time', 'priority']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in dataset")
        
        try:
            df['date_reported'] = pd.to_datetime(df['date_reported'], errors='coerce', infer_datetime_format=True)
            if df['date_reported'].isna().any():
                raise ValueError("Some date_reported values could not be converted to datetime")
        except Exception as e:
            raise ValueError(f"Failed to convert date_reported to datetime: {e}")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Trend analysis
def analyze_trends(df):
    """Analyze incident trends over time and by category."""
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['date_reported']):
            df['date_reported'] = pd.to_datetime(df['date_reported'], errors='coerce')
            if df['date_reported'].isna().any():
                raise ValueError("Invalid date_reported values after conversion")
        
        # Convert Period to string for plotting compatibility
        df['month'] = df['date_reported'].dt.to_period('M').astype(str)
        monthly_trends = df.groupby('month').size().reset_index(name='count')
        category_counts = df['category'].value_counts().reset_index(name='count')
        category_counts.columns = ['category', 'count']
        return monthly_trends, category_counts
    except Exception as e:
        st.error(f"Error in trend analysis: {e}")
        return None, None

# Predictive model for ticket priority
def train_priority_model(df):
    """Train a RandomForest model to predict ticket priority."""
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = tfidf.fit_transform(df['description'].fillna(''))
    X = pd.DataFrame(X_text.toarray(), columns=[f'text_{i}' for i in range(X_text.shape[1])])
    X['resolution_time'] = df['resolution_time'].fillna(df['resolution_time'].mean())
    
    y = df['priority'].map({'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Critical'])
    
    return model, tfidf, report

# Streamlit app
def main():
    st.title("Incident Trend Analysis and Predictive Ticket Prioritization")
    
    data_option = st.radio("Choose data source:", ("Upload CSV", "Generate Synthetic Data"))
    
    df = None
    if data_option == "Generate Synthetic Data":
        st.info("Generating synthetic data...")
        df, file_path = generate_synthetic_data()
        st.success(f"Synthetic data generated and saved as {file_path}")
        st.download_button("Download Synthetic Data", data=open(file_path, 'rb').read(), file_name=file_path)
    else:
        uploaded_file = st.file_uploader("Upload Incident Data (CSV)", type="csv")
        if uploaded_file:
            df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Incident Trends")
        monthly_trends, category_counts = analyze_trends(df)
        if monthly_trends is not None and not monthly_trends.empty:
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=monthly_trends, x='month', y='count')
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            plt.figure(figsize=(10, 5))
            sns.barplot(data=category_counts, x='category', y='count')
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.error("Unable to generate trend plots due to data issues.")
        
        st.subheader("Priority Prediction")
        model, tfidf, report = train_priority_model(df)
        st.text("Model Performance:")
        st.text(report)
        
        st.subheader("Predict Priority for New Ticket")
        description = st.text_area("Enter ticket description")
        resolution_time = st.number_input("Estimated resolution time (hours)", min_value=0.0)
        
        if st.button("Predict"):
            if description:
                input_text = tfidf.transform([description]).toarray()
                input_df = pd.DataFrame(input_text, columns=[f'text_{i}' for i in range(input_text.shape[1])])
                input_df['resolution_time'] = resolution_time
                prediction = model.predict(input_df)[0]
                priority_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
                st.success(f"Predicted Priority: {priority_map[prediction]}")
            else:
                st.error("Please enter a description")

if __name__ == "__main__":
    main()

