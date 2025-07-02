import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import random
import plotly.express as px # For better visualizations

# --- Global Definitions ---
# Define priorities globally so they are accessible throughout the script
priorities = ['Low', 'Medium', 'High', 'Critical']
priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
reverse_priority_mapping = {v: k for k, v in priority_mapping.items()}

# --- Functions for Core Logic ---

@st.cache_data # Cache the data simulation to avoid re-running on every interaction
def simulate_incident_data(num_records=1000):
    """
    Simulates a dataset of service desk incidents for demonstration purposes.
    """
    st.write(f"Simulating {num_records} incident records...")

    categories = ['Software', 'Hardware', 'Network', 'Security', 'Access']
    subcategories = {
        'Software': ['Application Crash', 'Performance Issue', 'Bug Report', 'Installation'],
        'Hardware': ['Laptop Issue', 'Desktop Issue', 'Printer Issue', 'Monitor Issue'],
        'Network': ['Connectivity Loss', 'Slow Network', 'VPN Issue'],
        'Security': ['Phishing Email', 'Account Compromise', 'Malware'],
        'Access': ['Password Reset', 'Account Lockout', 'New User Access']
    }
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed', 'On Hold']
    descriptions = [
        "Application XYZ is crashing frequently.",
        "My laptop is not turning on.",
        "Cannot connect to the office Wi-Fi.",
        "Received a suspicious email.",
        "Forgot my password.",
        "Printer is not responding.",
        "Slow performance on network drive.",
        "Need access to shared folder ABC.",
        "Software update failed to install.",
        "Monitor flickering intermittently."
    ]

    data = []
    start_date = datetime.now() - timedelta(days=365) # Data for the last year

    for i in range(num_records):
        incident_id = f"INC{10000 + i}"
        timestamp = start_date + timedelta(minutes=random.randint(0, 365 * 24 * 60))
        category = random.choice(categories)
        subcategory = random.choice(subcategories[category])
        description = random.choice(descriptions)
        priority = random.choice(priorities)
        status = random.choice(statuses)

        if priority == 'Critical':
            resolution_hours = random.randint(1, 8)
        elif priority == 'High':
            resolution_hours = random.randint(4, 24)
        elif priority == 'Medium':
            resolution_hours = random.randint(12, 72)
        else: # Low
            resolution_hours = random.randint(24, 168)

        resolution_time = timedelta(hours=resolution_hours)
        resolved_at = timestamp + resolution_time if status in ['Resolved', 'Closed'] else None

        data.append([incident_id, timestamp, category, subcategory, description,
                     priority, status, resolution_time.total_seconds() / 3600, resolved_at])

    df = pd.DataFrame(data, columns=[
        'incident_id', 'timestamp', 'category', 'subcategory', 'description',
        'priority', 'status', 'resolution_hours_actual', 'resolved_at'
    ])
    st.success("Data simulation complete!")
    return df

@st.cache_resource # Cache the model and encoders as they are resources
def train_model(df_raw):
    """
    Preprocesses data, trains the RandomForestClassifier model, and returns
    the model, label encoders, and evaluation metrics.
    """
    st.write("--- Starting Data Preprocessing & Model Training ---")

    df = df_raw.copy() # Work on a copy to avoid modifying the original cached DataFrame

    # Convert 'timestamp' to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

    df.loc[:, 'actual_resolution_hours'] = df['resolution_hours_actual'].fillna(0)

    # Encode categorical variables for machine learning
    label_encoders = {}
    for column in ['category', 'subcategory', 'status']:
        le = LabelEncoder()
        df[column + '_encoded'] = le.fit_transform(df[column])
        label_encoders[column] = le

    df['priority_encoded'] = df['priority'].map(priority_mapping)

    ml_df = df.drop(columns=[
        'incident_id', 'timestamp', 'description', 'priority', 'status',
        'resolution_hours_actual', 'resolved_at'
    ])
    ml_df.dropna(inplace=True)

    st.write("Data preprocessing complete.")
    st.write(f"Shape of ML-ready data: {ml_df.shape}")

    # Define features (X) and target (y)
    features = [
        'category_encoded', 'subcategory_encoded', 'status_encoded',
        'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'week_of_year'
    ]
    X = ml_df[features]
    y = ml_df['priority_encoded']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.write(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Initialize and train the RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    st.write("RandomForestClassifier model trained successfully.")

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test.map(reverse_priority_mapping),
                                   pd.Series(y_pred).map(reverse_priority_mapping),
                                   target_names=priorities, output_dict=True)

    st.success("Model training and evaluation complete!")
    return model, label_encoders, accuracy, report, df # Return the processed df for trend analysis

# --- Streamlit Application Layout ---

st.set_page_config(layout="wide", page_title="Incident Analysis & Priority Prediction")

st.title("📊 Incident Trend Analysis & Predictive Ticket Prioritization")
st.markdown("""
This application demonstrates how to analyze service desk incident trends and predict ticket priority
using Python, Pandas, and Scikit-learn.
""")

# --- Sidebar for Data Simulation ---
st.sidebar.header("Data Simulation")
num_records_sim = st.sidebar.slider("Number of Incident Records to Simulate", 500, 5000, 2000, 100)
if st.sidebar.button("Simulate Incident Data & Train Model"):
    st.session_state['df_raw'] = simulate_incident_data(num_records_sim)
    model, label_encoders, accuracy, report, df_processed = train_model(st.session_state['df_raw'])
    st.session_state['model'] = model
    st.session_state['label_encoders'] = label_encoders
    st.session_state['accuracy'] = accuracy
    st.session_state['classification_report'] = report
    st.session_state['df_processed_for_trends'] = df_processed # Store processed df for trends
    st.sidebar.success("Data simulated and model trained!")

# Check if data is available in session state
if 'df_raw' not in st.session_state:
    st.info("Please simulate data using the sidebar button to proceed.")
else:
    df_processed = st.session_state['df_processed_for_trends']
    model = st.session_state['model']
    label_encoders = st.session_state['label_encoders']
    accuracy = st.session_state['accuracy']
    report = st.session_state['classification_report']

    st.header("📈 Incident Trend Analysis")
    st.markdown("Explore patterns and insights from the historical incident data.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Incidents per Day of Week")
        incidents_per_day = df_processed['day_of_week'].value_counts().sort_index()
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        incidents_per_day.index = [day_names[i] for i in incidents_per_day.index]
        fig_day = px.bar(incidents_per_day, x=incidents_per_day.index, y='count',
                         labels={'x':'Day of Week', 'y':'Number of Incidents'},
                         title='Incidents by Day of Week')
        st.plotly_chart(fig_day, use_container_width=True)

    with col2:
        st.subheader("Incidents per Hour of Day")
        incidents_per_hour = df_processed['hour_of_day'].value_counts().sort_index()
        fig_hour = px.bar(incidents_per_hour, x=incidents_per_hour.index, y='count',
                          labels={'x':'Hour of Day', 'y':'Number of Incidents'},
                          title='Incidents by Hour of Day')
        st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Top Incident Categories")
    top_categories = df_processed['category'].value_counts()
    fig_cat = px.pie(top_categories, values='count', names=top_categories.index,
                     title='Distribution of Incident Categories')
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Average Resolution Time (Hours) by Priority")
    avg_resolution_by_priority = df_processed.groupby('priority')['actual_resolution_hours'].mean().reindex(priorities)
    fig_res = px.bar(avg_resolution_by_priority, x=avg_resolution_by_priority.index, y='actual_resolution_hours',
                     labels={'x':'Priority', 'y':'Average Resolution Time (Hours)'},
                     title='Average Resolution Time by Priority')
    st.plotly_chart(fig_res, use_container_width=True)

    st.subheader("Incidents by Category and Priority")
    incidents_by_cat_priority = df_processed.groupby(['category', 'priority']).size().unstack(fill_value=0)
    st.dataframe(incidents_by_cat_priority)

    st.markdown("---")

    st.header("🤖 Predictive Ticket Prioritization")
    st.markdown("Enter details for a new incident to predict its priority.")

    # Get unique values for dropdowns from the processed dataframe
    unique_categories = df_processed['category'].unique().tolist()
    unique_subcategories = df_processed['subcategory'].unique().tolist()
    unique_statuses = df_processed['status'].unique().tolist()

    with st.form("new_incident_form"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            new_category = st.selectbox("Category", unique_categories)
            new_status = st.selectbox("Status", unique_statuses)
        with col_in2:
            # Filter subcategories based on selected category (more complex, for simplicity, show all)
            # A more advanced implementation would filter subcategories dynamically.
            new_subcategory = st.selectbox("Subcategory", unique_subcategories)
        new_description = st.text_area("Description (Optional, not used by current model)")

        predict_button = st.form_submit_button("Predict Priority")

        if predict_button:
            # Preprocess the new incident data
            new_incident_data = {
                'category': new_category,
                'subcategory': new_subcategory,
                'status': new_status,
                'timestamp': datetime.now() # Use current time for prediction
            }
            new_incident_df = pd.DataFrame([new_incident_data])
            new_incident_df['timestamp'] = pd.to_datetime(new_incident_df['timestamp'])

            new_incident_df['hour_of_day'] = new_incident_df['timestamp'].dt.hour
            new_incident_df['day_of_week'] = new_incident_df['timestamp'].dt.dayofweek
            new_incident_df['month'] = new_incident_df['timestamp'].dt.month
            new_incident_df['day_of_month'] = new_incident_df['timestamp'].dt.day
            new_incident_df['week_of_year'] = new_incident_df['timestamp'].dt.isocalendar().week.astype(int)

            # Apply the same label encoders
            features_for_prediction = [
                'category_encoded', 'subcategory_encoded', 'status_encoded',
                'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'week_of_year'
            ]
            processed_input = pd.DataFrame(columns=features_for_prediction)

            for column in ['category', 'subcategory', 'status']:
                try:
                    processed_input[column + '_encoded'] = label_encoders[column].transform(new_incident_df[column])
                except ValueError:
                    st.warning(f"Unknown value '{new_incident_df[column].iloc[0]}' for {column}. This might affect prediction accuracy.")
                    # Fallback for unknown values: use a default or handle as an error
                    # For simplicity, we'll assign a placeholder, but in a real app,
                    # you'd need a more robust strategy (e.g., OHE with handle_unknown='ignore')
                    processed_input[column + '_encoded'] = -1 # Or a specific 'unknown' category if encoded

            # Add time-based features
            processed_input['hour_of_day'] = new_incident_df['hour_of_day']
            processed_input['day_of_week'] = new_incident_df['day_of_week']
            processed_input['month'] = new_incident_df['month']
            processed_input['day_of_month'] = new_incident_df['day_of_month']
            processed_input['week_of_year'] = new_incident_df['week_of_year']

            # Ensure column order matches training data
            processed_input = processed_input[features_for_prediction]

            # Make prediction
            try:
                predicted_priority_encoded = model.predict(processed_input)
                predicted_priority_label = reverse_priority_mapping[predicted_priority_encoded[0]]
                st.success(f"**Predicted Priority: {predicted_priority_label}**")
            except Exception as e:
                st.error(f"Error during prediction: {e}. Please ensure all inputs are valid.")

    st.markdown("---")

    st.header("📊 Model Performance")
    st.write(f"Overall Accuracy: **{accuracy:.2f}**")
    st.subheader("Classification Report")
    # Display the classification report as a DataFrame for better readability
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.markdown("---")
    st.markdown("Project by: Venkateswarlu Meruva")
