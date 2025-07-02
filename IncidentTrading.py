import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from datetime import datetime, timedelta
import random
import plotly.express as px

# --- Global Definitions ---
# Updated priorities to P1-P4
priorities = ['P4', 'P3', 'P2', 'P1'] # P4 (Low), P3 (Medium), P2 (High), P1 (Critical)
priority_mapping = {'P4': 0, 'P3': 1, 'P2': 2, 'P1': 3}
reverse_priority_mapping = {v: k for k, v in priority_mapping.items()}

# --- Functions for Core Logic ---

@st.cache_data # Cache the data simulation to avoid re-running on every interaction
def simulate_incident_data(num_records=2000):
    """
    Simulates a dataset of service desk incidents for demonstration purposes.
    Now includes 'impact' and uses P1-P4 priorities.
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
    impacts = ['Low', 'Medium', 'High'] # New 'impact' levels
    statuses = ['Open', 'In Progress', 'Resolved', 'Closed', 'On Hold']
    descriptions = [
        "Application XYZ is crashing frequently and affecting multiple users.",
        "My laptop is not turning on, critical for presentation tomorrow.",
        "Cannot connect to the office Wi-Fi, impacting entire department.",
        "Received a suspicious email with ransomware attachment, potential security breach.",
        "Forgot my password, cannot access any systems.",
        "Printer in finance department is not responding, urgent report stuck.",
        "Slow performance on network drive, affecting all team members.",
        "Need access to shared folder ABC, blocking project progress.",
        "Software update failed to install, causing system instability.",
        "Monitor flickering intermittently, causing eye strain.",
        "Email server is down, no one can send or receive emails.",
        "Database is unresponsive, critical business application is offline.",
        "VPN connection dropping frequently for remote workers.",
        "Unauthorized access attempt detected on server.",
        "New user account creation failed, preventing onboarding.",
        "Payroll system is showing incorrect data, high impact.",
        "Internet outage in building C, affecting 50+ employees.",
        "Specific application module is not loading for a single user.",
        "Mouse not working on my workstation.",
        "Request for new software installation on my machine."
    ]

    data = []
    start_date = datetime.now() - timedelta(days=365) # Data for the last year

    for i in range(num_records):
        incident_id = f"INC{10000 + i}"
        timestamp = start_date + timedelta(minutes=random.randint(0, 365 * 24 * 60))
        category = random.choice(categories)
        subcategory = random.choice(subcategories[category])
        description = random.choice(descriptions)
        impact = random.choice(impacts) # Assign impact
        priority = random.choice(priorities) # Use P1-P4 priorities
        status = random.choice(statuses)

        # Simulate resolution time based on priority (P1 faster, P4 slower)
        if priority == 'P1': # Critical
            resolution_hours = random.randint(1, 8)
        elif priority == 'P2': # High
            resolution_hours = random.randint(4, 24)
        elif priority == 'P3': # Medium
            resolution_hours = random.randint(12, 72)
        else: # P4 (Low)
            resolution_hours = random.randint(24, 168)

        resolution_time = timedelta(hours=resolution_hours)
        resolved_at = timestamp + resolution_time if status in ['Resolved', 'Closed'] else None

        data.append([incident_id, timestamp, category, subcategory, description,
                     impact, priority, status, resolution_time.total_seconds() / 3600, resolved_at])

    df = pd.DataFrame(data, columns=[
        'incident_id', 'timestamp', 'category', 'subcategory', 'description',
        'impact', 'priority', 'status', 'resolution_hours_actual', 'resolved_at'
    ])
    st.success("Data simulation complete!")
    return df

@st.cache_resource # Cache the model and encoders as they are resources
def train_model(df_raw):
    """
    Preprocesses data (including NLP for description), trains the LogisticRegression model,
    and returns the model, label encoders, TF-IDF vectorizer, and evaluation metrics.
    Handles potential missing 'resolved_at'/'resolution_hours_actual' for uploaded data.
    """
    st.write("--- Starting Data Preprocessing & Model Training ---")

    df = df_raw.copy() # Work on a copy to avoid modifying the original cached DataFrame

    # --- Data Validation for essential columns ---
    required_cols = ['timestamp', 'category', 'subcategory', 'description', 'impact', 'priority', 'status']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please ensure your CSV has these columns.")
        return None, None, None, None, None, None # Return None for all if validation fails

    # Convert 'timestamp' to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') # Coerce errors to NaT
    df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed

    # Extract time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

    # Handle 'actual_resolution_hours' for trend analysis
    if 'resolution_hours_actual' in df.columns:
        df.loc[:, 'actual_resolution_hours'] = df['resolution_hours_actual'].fillna(0)
    elif 'resolved_at' in df.columns:
        df['resolved_at'] = pd.to_datetime(df['resolved_at'], errors='coerce')
        df.loc[:, 'actual_resolution_hours'] = (df['resolved_at'] - df['timestamp']).dt.total_seconds() / 3600
        df.loc[:, 'actual_resolution_hours'] = df['actual_resolution_hours'].fillna(0)
    else:
        st.warning("Neither 'resolution_hours_actual' nor 'resolved_at' found. Average resolution time trends will not be available.")
        df.loc[:, 'actual_resolution_hours'] = 0 # Default to 0 if not present

    # Encode categorical variables
    label_encoders = {}
    for column in ['category', 'subcategory', 'impact', 'status']:
        le = LabelEncoder()
        # Handle potential new categories in uploaded data by fitting on unique values
        df[column + '_encoded'] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Map priority to numerical values
    # Handle cases where priority labels might be different in uploaded CSV
    df['priority_encoded'] = df['priority'].map(priority_mapping)
    # If there are NaN values after mapping (due to unknown priority labels), drop them or handle them
    df.dropna(subset=['priority_encoded'], inplace=True)
    df['priority_encoded'] = df['priority_encoded'].astype(int) # Ensure integer type

    # --- NLP for Description ---
    # Ensure description column is string type and fill NaNs
    df['description'] = df['description'].astype(str).fillna('')
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(df['description'])
    st.write(f"TF-IDF vectorizer fitted. Number of text features: {X_tfidf.shape[1]}")

    # Prepare other features
    numerical_features = [
        'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'week_of_year'
    ]
    encoded_categorical_features = [
        'category_encoded', 'subcategory_encoded', 'impact_encoded', 'status_encoded'
    ]

    # Ensure these columns exist before selecting
    for col in numerical_features + encoded_categorical_features:
        if col not in df.columns:
            st.error(f"Missing processed feature column: {col}. This indicates an issue with preprocessing.")
            return None, None, None, None, None, None

    X_other_features = df[numerical_features + encoded_categorical_features].values

    # Combine TF-IDF features with other features
    X = hstack([X_tfidf, X_other_features])
    y = df['priority_encoded']

    st.write(f"Combined feature matrix shape: {X.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.write(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Initialize and train the LogisticRegression model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)
    st.write("LogisticRegression model trained successfully.")

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test.map(reverse_priority_mapping),
                                   pd.Series(y_pred).map(reverse_priority_mapping),
                                   target_names=priorities, output_dict=True)

    st.success("Model training and evaluation complete!")
    return model, label_encoders, tfidf_vectorizer, accuracy, report, df # Return processed df for trend analysis

# --- Streamlit Application Layout ---

st.set_page_config(layout="wide", page_title="Incident Analysis & Priority Prediction")

st.title("📊 Incident Trend Analysis & Predictive Ticket Prioritization")
st.markdown("""
This application analyzes service desk incident trends and predicts ticket priority (P1-P4)
using Python, Pandas, Scikit-learn (Logistic Regression, TF-IDF), and Streamlit.
""")

# --- Sidebar for Data Input and Training ---
st.sidebar.header("Data Input & Model Training")
data_source = st.sidebar.radio("Choose Data Source:", ("Simulate Data", "Upload CSV"))

df_raw_uploaded = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df_raw_uploaded = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file uploaded successfully!")
            st.sidebar.dataframe(df_raw_uploaded.head())
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            df_raw_uploaded = None

if st.sidebar.button("Process Data & Train Model"):
    if data_source == "Simulate Data":
        num_records_sim = st.sidebar.slider("Number of Incident Records to Simulate", 1000, 5000, 2000, 100)
        st.session_state['df_raw'] = simulate_incident_data(num_records_sim)
    elif data_source == "Upload CSV" and df_raw_uploaded is not None:
        st.session_state['df_raw'] = df_raw_uploaded
    else:
        st.sidebar.warning("Please upload a CSV file or select 'Simulate Data'.")
        st.session_state['df_raw'] = None

    if st.session_state.get('df_raw') is not None:
        model, label_encoders, tfidf_vectorizer, accuracy, report, df_processed = train_model(st.session_state['df_raw'])
        if model: # Check if model training was successful (i.e., no validation errors)
            st.session_state['model'] = model
            st.session_state['label_encoders'] = label_encoders
            st.session_state['tfidf_vectorizer'] = tfidf_vectorizer
            st.session_state['accuracy'] = accuracy
            st.session_state['classification_report'] = report
            st.session_state['df_processed_for_trends'] = df_processed
            st.sidebar.success("Data processed and model trained!")
        else:
            st.sidebar.error("Model training failed due to data issues. Please check your CSV or simulated data.")
            # Clear session state for model related items if training failed
            st.session_state.pop('model', None)
            st.session_state.pop('label_encoders', None)
            st.session_state.pop('tfidf_vectorizer', None)
            st.session_state.pop('accuracy', None)
            st.session_state.pop('classification_report', None)
            st.session_state.pop('df_processed_for_trends', None)


# Check if data and model are available in session state
if 'df_raw' not in st.session_state or 'model' not in st.session_state:
    st.info("Please select a data source and click 'Process Data & Train Model' in the sidebar to proceed.")
else:
    df_processed = st.session_state['df_processed_for_trends']
    model = st.session_state['model']
    label_encoders = st.session_state['label_encoders']
    tfidf_vectorizer = st.session_state['tfidf_vectorizer']
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
    # Only show if actual_resolution_hours has meaningful data
    if df_processed['actual_resolution_hours'].sum() > 0:
        avg_resolution_by_priority = df_processed.groupby('priority')['actual_resolution_hours'].mean().reindex(priorities)
        fig_res = px.bar(avg_resolution_by_priority, x=avg_resolution_by_priority.index, y='actual_resolution_hours',
                         labels={'x':'Priority', 'y':'Average Resolution Time (Hours)'},
                         title='Average Resolution Time by Priority',
                         category_orders={"x": priorities}) # Ensure P1-P4 order
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Average Resolution Time chart not available as 'resolution_hours_actual' or 'resolved_at' column was not found or contained no data.")


    st.subheader("Incidents by Category and Priority")
    incidents_by_cat_priority = df_processed.groupby(['category', 'priority']).size().unstack(fill_value=0)
    # Reorder columns to match P1-P4
    incidents_by_cat_priority = incidents_by_cat_priority[priorities] if all(p in incidents_by_cat_priority.columns for p in priorities) else incidents_by_cat_priority
    st.dataframe(incidents_by_cat_priority)

    st.markdown("---")

    st.header("🤖 Predictive Ticket Prioritization")
    st.markdown("Enter details for a new incident to predict its priority.")

    # Get unique values for dropdowns from the processed dataframe
    unique_categories = df_processed['category'].unique().tolist()
    unique_subcategories = df_processed['subcategory'].unique().tolist()
    unique_impacts = df_processed['impact'].unique().tolist()
    unique_statuses = df_processed['status'].unique().tolist()

    with st.form("new_incident_form"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            new_category = st.selectbox("Category", unique_categories)
            new_impact = st.selectbox("Impact", unique_impacts)
        with col_in2:
            new_subcategory = st.selectbox("Subcategory", unique_subcategories)
            new_status = st.selectbox("Status", unique_statuses)
        new_description = st.text_area("Description", "My application is crashing and affecting multiple users.")

        predict_button = st.form_submit_button("Predict Priority")

        if predict_button:
            # Preprocess the new incident data
            new_incident_data = {
                'category': new_category,
                'subcategory': new_subcategory,
                'impact': new_impact,
                'status': new_status,
                'description': new_description,
                'timestamp': datetime.now() # Use current time for prediction
            }
            new_incident_df = pd.DataFrame([new_incident_data])
            new_incident_df['timestamp'] = pd.to_datetime(new_incident_df['timestamp'])

            new_incident_df['hour_of_day'] = new_incident_df['timestamp'].dt.hour
            new_incident_df['day_of_week'] = new_incident_df['timestamp'].dt.dayofweek
            new_incident_df['month'] = new_incident_df['timestamp'].dt.month
            new_incident_df['day_of_month'] = new_incident_df['timestamp'].dt.day
            new_incident_df['week_of_year'] = new_incident_df['timestamp'].dt.isocalendar().week.astype(int)

            # Apply label encoders and TF-IDF vectorizer
            processed_input_numerical_categorical = pd.DataFrame()
            for column in ['category', 'subcategory', 'impact', 'status']:
                try:
                    processed_input_numerical_categorical[column + '_encoded'] = label_encoders[column].transform(new_incident_df[column])
                except ValueError:
                    st.warning(f"Unknown value '{new_incident_df[column].iloc[0]}' for {column}. This might affect prediction accuracy.")
                    # Assign a default value if unknown, e.g., 0 or a specific 'unknown' category if encoded
                    # For simplicity, if a value is unknown, we'll assign 0. This might not be ideal
                    # for all models but prevents immediate crash.
                    processed_input_numerical_categorical[column + '_encoded'] = 0

            # Add time-based features
            processed_input_numerical_categorical['hour_of_day'] = new_incident_df['hour_of_day']
            processed_input_numerical_categorical['day_of_week'] = new_incident_df['day_of_week']
            processed_input_numerical_categorical['month'] = new_incident_df['month']
            processed_input_numerical_categorical['day_of_month'] = new_incident_df['day_of_month']
            processed_input_numerical_categorical['week_of_year'] = new_incident_df['week_of_year']

            # Transform new description using the fitted TF-IDF vectorizer
            # Ensure description is string and fillna for vectorizer
            new_description_tfidf = tfidf_vectorizer.transform([new_description])

            # Combine all features for prediction
            numerical_features_order = [
                'hour_of_day', 'day_of_week', 'month', 'day_of_month', 'week_of_year'
            ]
            encoded_categorical_features_order = [
                'category_encoded', 'subcategory_encoded', 'impact_encoded', 'status_encoded'
            ]
            combined_features_order = numerical_features_order + encoded_categorical_features_order

            # Convert processed_input_numerical_categorical to a sparse matrix for hstack
            processed_input_sparse = processed_input_numerical_categorical[combined_features_order].sparse.to_coo()

            # Ensure the number of columns from TF-IDF and other features match the training data
            # This is critical if max_features was set for TF-IDF
            # If the number of columns don't match, hstack will raise an error.
            # We need to ensure the number of columns in processed_input_sparse matches
            # the number of non-TFIDF columns in the training data.
            # The easiest way to ensure this is to create a dummy DataFrame with all features
            # and then fill it with the actual values.

            # Create a dummy array with the correct number of non-TFIDF features
            # This is based on the `X_other_features` shape from training
            num_other_features_trained = model.coef_.shape[1] - tfidf_vectorizer.vocabulary_.__len__()
            if num_other_features_trained < 0: # This means TFIDF features are more than total features, which is wrong
                st.error("Error: Mismatch in feature counts. TF-IDF features might be too many or other features too few.")
                final_prediction_input = None
            else:
                # Create a dense array for the non-TFIDF features, then convert to sparse
                other_features_array = np.zeros((1, len(combined_features_order)))
                for i, col in enumerate(combined_features_order):
                    if col in processed_input_numerical_categorical.columns:
                        other_features_array[0, i] = processed_input_numerical_categorical[col].iloc[0]

                # Convert to sparse format
                other_features_sparse = pd.DataFrame(other_features_array, columns=combined_features_order).sparse.to_coo()

                # Ensure new_description_tfidf has the same number of columns as during training
                # This is handled by TfidfVectorizer's transform method if it was fitted correctly.
                # However, if the vocabulary is empty or very small, it might cause issues.
                if new_description_tfidf.shape[1] != tfidf_vectorizer.vocabulary_.__len__():
                     st.warning("TF-IDF features mismatch. This might indicate an issue with the description or vectorizer.")
                     # Attempt to resize the TF-IDF matrix if possible, or pad with zeros
                     # For robustness, ensure new_description_tfidf has the same number of columns as the original X_tfidf
                     # This is usually handled by transform, but if the vocabulary is empty, it can be 0.
                     # If the vocabulary is empty, new_description_tfidf.shape[1] will be 0.
                     # We need it to be tfidf_vectorizer.max_features or actual vocabulary size.
                     # A more robust approach would be to pad/truncate if mismatch, but TFIDF handles this.
                     pass # TfidfVectorizer.transform should handle this if fitted correctly.

                final_prediction_input = hstack([new_description_tfidf, other_features_sparse])

            # Make prediction
            if final_prediction_input is not None:
                try:
                    predicted_priority_encoded = model.predict(final_prediction_input)
                    predicted_priority_label = reverse_priority_mapping[predicted_priority_encoded[0]]
                    st.success(f"**Predicted Priority: {predicted_priority_label}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}. Please ensure all inputs are valid and model is trained.")
            else:
                st.error("Prediction input could not be formed due to feature mismatch.")

    st.markdown("---")

    st.header("📊 Model Performance")
    st.write(f"Overall Accuracy: **{accuracy:.2f}**")
    st.subheader("Classification Report")
    # Display the classification report as a DataFrame for better readability
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.markdown("---")
    st.markdown("Project by: Venkateswarlu Meruva")
