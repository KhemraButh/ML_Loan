import streamlit as st
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.exc import SQLAlchemyError
import joblib
import os
import pandas as pd
from sqlalchemy import text # Import text for explicit SQL
import matplotlib.pyplot as plt
import requests
import dice_ml
from dice_ml import Dice

# -------- Session State for Login --------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Initialize last_application_id_generated for consistent display
if 'last_application_id_generated' not in st.session_state:
    st.session_state.last_application_id_generated = "N/A"

# -------- Page Config --------
st.set_page_config(page_title="Loan Approval System", layout="wide")

# -------- CSS Styling --------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        .header {
            background-color: #e6f4ea;
            padding: 1em;
            border-radius: 12px;
            text-align: center;
            color: #006400;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 1em;
        }
        .card {
            background-color: white;
            padding: 1.5em;
            border-radius: 12px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1em;
        }
        .table-container {
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 15px;
            margin-bottom: 1em;
        }
        .table {
            width: 100%;
            border-collapse: collapse;
        }
        .table th, .table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .table th {
            background-color: #008A04;
            color: white;
        }
        div.stButton > button:first-child {
            background-color: #008A04;
            color: white;
            border-radius: 8px;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Database connection
engine = create_engine('postgresql://postgres:TheK@localhost:5433/postgres')
from sqlalchemy import inspect


#inspector = inspect(connection)
def get_next_application_id():
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            table_exists = 'loancamdata' in inspector.get_table_names()

            if not table_exists:
                return "KH00001"

            result = connection.execute(text("""
                SELECT "Application_ID" 
                FROM loancamdata
                ORDER BY CAST(SUBSTRING("Application_ID", 3) AS INTEGER) DESC 
                LIMIT 1
            """)).scalar_one_or_none()

            if result:
                last_id_num = int(result[2:])
                next_id_num = last_id_num + 1
                return f"KH{next_id_num:05d}"
            else:
                return "KH00001"
    except Exception as e:
        st.error(f"Error getting Application ID: {e}")
        return "KH00001"

def calculate_derived_fields(data):
    """Calculate DTI and LVR before saving to database"""
    try:
        data['Dti'] = (data['Debt'] / data['Income']) * 100 if data['Income'] > 0 else 0
    except ZeroDivisionError:
        data['Dti'] = 0
    except Exception:
        data['Dti'] = 0

    try:
        data['LVR'] = (data['Loan Amount'] / data['Collateral Value']) * 100 if data['Collateral Value'] > 0 else 0
    except ZeroDivisionError:
        data['LVR'] = 0
    except Exception:
        data['LVR'] = 0
    return data


# Define model paths
BASE_DIR = os.path.dirname(__file__)  # folder where app.py lives

MODEL_PATHS = {
    'Home Loan': os.path.join(BASE_DIR, "frontend", "hl_pipeline.pkl"),
    'SME Loan': os.path.join(BASE_DIR, "frontend", "sme_pipeline.pkl"),
    'Personal Loan': os.path.join(BASE_DIR, "frontend", "person_pipeline.pkl"),
}

# Load all models at startup
for loan_type, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        st.error(f"{loan_type} model not found at {path}")
        st.stop()
    try:
        # Standardized model key format: "home_loan_model", "sme_loan_model", etc.
        model_key = f"{loan_type.lower().replace(' ', '_')}_model"
        st.session_state[model_key] = joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {loan_type} model: {str(e)}")
        st.stop()

def predict_loan_approval(form_data, loan_type):
    """Run prediction based on loan type with consistent model naming"""
    try:
        loan_type = form_data['loan_type']  # Expected values: 'Home Loan', 'SME Loan', 'Personal Loan'
        
        # Standardize model key (matches how we stored in session_state)
        model_key = f"{loan_type.lower().replace(' ', '_')}_model"
        model = st.session_state.get(model_key)
        
        if not model:
            raise ValueError(f"No model loaded for {loan_type}")
        
        # Feature preparation
        if loan_type == 'Personal Loan':
            input_data = pd.DataFrame([{
                'Age': form_data['customer_age'],
                'Income': form_data['income'],
                'Dti': (form_data['debt'] / form_data['income']) * 100 if form_data['income'] > 0 else 0,
                'Interest Rate': form_data['interest_rate'],
                'Loan Amount': form_data['loan_amount'],
                'Credit History': int(form_data['credit_history']),
                'Guarantor Binary': 1 if form_data['guarantor'] == 'Yes' else 0,
                'Employee_Status': form_data['employee_status'],
                'Customer Status': form_data['customer_status']
            }])
        
        elif loan_type == 'Home Loan':
            input_data = pd.DataFrame([{
                'Income': form_data['income'],
                'Dti': (form_data['debt'] / form_data['income']) * 100 if form_data['income'] > 0 else 0,
                'LVR': (form_data['loan_amount'] / form_data['collateral_value']) * 100 if form_data['collateral_value'] > 0 else 0,
                'Interest Rate': form_data['interest_rate'],
                'Credit History': int(form_data['credit_history']),
                'Guarantor Binary': 1 if form_data['guarantor'] == 'Yes' else 0,
                'Employee_Status': form_data['employee_status'],
                'Loan Term': form_data['loan_term'],
                'Collateral Type': form_data['collateral_type']
            }])
        
        elif loan_type == 'SME Loan':
            input_data = pd.DataFrame([{
                'Dti': (form_data['debt'] / form_data['income']) * 100 if form_data['income'] > 0 else 0,
                'Interest Rate': form_data['interest_rate'],
                'LVR': (form_data['loan_amount'] / form_data['collateral_value']) * 100 if form_data['collateral_value'] > 0 else 0,
                'Credit History': int(form_data['credit_history']),
                'Guarantor Binary': 1 if form_data['guarantor'] == 'Yes' else 0,
                'Employee_Status': form_data['employee_status'],
                'Loan Term': form_data['loan_term']
            }])
        
        else:
            raise ValueError(f"Unsupported loan type: {loan_type}")

        # Get prediction
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]

        return {
            'prediction': 'Approved' if prediction[0] == 1 else 'Rejected',
            'confidence': round(max(proba) * 100, 2),
            'loan_type': loan_type,
            'model_used': model_key,
            'features': input_data.iloc[0].to_dict()
        }
    except Exception as e:
        st.error(f"Prediction failed for {loan_type}: {str(e)}")
        return None

def save_to_database(form_data):
    """Prepare and save form data to PostgreSQL"""
    
    # Generate the Application ID
    application_id = get_next_application_id()
    if application_id is None:
        st.error("Could not generate a unique Application ID. Please try again.")
        return False
    
    result = predict_loan_approval(form_data, form_data['loan_type'])
    if result is None:
        st.error("Prediction failed. Data not saved.")
        return False

    prediction_label = result['prediction']
    confidence = result['confidence']  # Already in 0–100 scale

    # Set Loan Status based on confidence
    if confidence >= 80:
        loan_status = prediction_label  # "Approved" or "Rejected"
    else:
        loan_status = "Pending"

    db_data = {
        'Application_ID': application_id,
        'Full Name': form_data['customer_name'],
        'Age': form_data['customer_age'],
        'Location': form_data['customer_location'],
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'Loan Type': form_data['loan_type'],
        'Marital Status': form_data['marital_status'],
        'Collateral Type': form_data['collateral_type'],
        'Collateral Value': form_data['collateral_value'],
        'Employee_Status': form_data['employee_status'],
        'Income': form_data['income'],
        'Loan Term': form_data['loan_term'],
        'Loan Amount': form_data['loan_amount'],
        'Debt': form_data['debt'],
        'Guarantor': form_data['guarantor'],
        'Customer Status': form_data['customer_status'],
        'Customer Relation': form_data['customer_relation_bank'],
        'Interest Rate': form_data['interest_rate'],
        'Loan Reason': form_data['loan_reason'],
        'Credit History': form_data['credit_history'],
        'Loan Status': loan_status,
        'RM_Code': form_data['rm_code'],
        'id_card': form_data['id_card'],
        'land_plan': form_data['land_plan']
    }

    # Calculate derived fields
    db_data = calculate_derived_fields(db_data)

    # Convert to DataFrame
    df = pd.DataFrame([db_data])

    try:
        df.to_sql('loancamdata', engine, if_exists='append', index=False)
        st.session_state.last_application_id_generated = application_id # Store for display
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
def get_recent_applications(limit=100):
    """Fetch recent loan applications from database"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    "Application_ID" as customer_id,
                    "Full Name" as customer_name,
                    "Loan Amount",
                    "Loan Type",
                    "Credit History",
                    "Dti",
                    "LVR",
                    "Loan Status" as approval_status,
                    "Date" as application_date
                FROM loancamdata
                ORDER BY "Application_ID" DESC
                LIMIT :limit
            """)
            result = conn.execute(query, {'limit': limit})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    
from sqlalchemy.exc import SQLAlchemyError

def update_decision(application_id, decision):
    """Update the final decision in database (and override ML decision in 'Loan Status')"""
    try:
        with engine.begin() as conn:
            query = text("""
                UPDATE loancamdata
                SET "Loan Status" = :decision
                WHERE "Application_ID" = :app_id
            """)
            conn.execute(query, {'decision': decision, 'app_id': application_id})
        return True
    except SQLAlchemyError as e:
        st.error(f"❌ Failed to update decision:\n\n{str(e)}")
        return False

def show_approval_dashboard():
    # Header Section
    col1, col2 = st.columns([1, 4])
    with col1:
        image_path = os.path.join(BASE_DIR, "frontend", "Logo-CMCB.png")
        st.image(image_path, width=100)

    with col2:
        st.markdown("<h1 style='text-align: left; color: green;'>Approval Dashboard for Risk Team</h1>", 
                    unsafe_allow_html=True)
    # Filters Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        loan_type_filter = st.selectbox(
            "Loan Type",
            ["All"] + list(get_recent_applications(1)['Loan Type'].unique()),
            index=0
        )
        status_filter = st.selectbox(
            "Approval Status",
            ["All", "Approved", "Rejected", "Pending"],
            index=0
        )
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Get summary stats
        df = get_recent_applications()
        if not df.empty:
            total_apps = len(df)
            approved = len(df[df['approval_status'] == 'Approved'])
            rejected = len(df[df['approval_status'] == 'Rejected'])
            
            st.metric("Total Applications", total_apps)
            st.metric("Approval Rate", f"{(approved/total_apps)*100:.1f}%")
            st.metric("Rejection Rate", f"{(rejected/total_apps)*100:.1f}%")

    # Main Content Area
    st.markdown("### 📝 Recent Loan Applications")
    
    # Get data from database
    df = get_recent_applications(100)  # Get last 100 applications
    
    if df.empty:
        st.warning("No loan applications found in database")
        return
    
    # Apply filters
    if loan_type_filter != "All":
        df = df[df['Loan Type'] == loan_type_filter]
    if status_filter != "All":
        df = df[df['approval_status'] == status_filter]
    
    # Add decision tracking columns if not present
    if 'risk_team_decision' not in df.columns:
        df['risk_team_decision'] = "Pending"
    if 'final_decision' not in df.columns:
        df['final_decision'] = df['approval_status']  # Initialize with current status
    
    # Format the display DataFrame
    display_df = df[[
        'customer_id', 'customer_name', 'Loan Type', 'Loan Amount',
        'approval_status', 'risk_team_decision', 'final_decision'
    ]].copy()
    
    # Style the DataFrame
    def color_status(val):
        if val == 'Approved':
            return 'background-color: #4CAF50; color: white;'
        elif val == 'Rejected':
            return 'background-color: #F44336; color: white;'
        else:
            return 'background-color: #FF9800; color: white;'
    
    styled_df = display_df.style.map(color_status, subset=['approval_status', 'final_decision'])

    #styled_df = display_df.style.applymap(color_status, subset=['approval_status', 'final_decision'])
    
    # Display the styled DataFrame
    st.dataframe(
        styled_df,
        column_config={
            "customer_id": "App ID",
            "customer_name": "Customer",
            "Loan Amount": st.column_config.NumberColumn(format="$%.2f"),
            "approval_status": "Model Decision",
            "risk_team_decision": "Risk Team",
            "final_decision": "Final Status"
        },
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Decision Making Section
    st.markdown("### 🖊️ Decision Panel")
    
    if not df.empty:
        # When selecting
        selected_id = st.selectbox(
            "Select Application ID",
            df['customer_id'].unique(),  # Use actual DB column name
            format_func=lambda x: f"{x} - {df[df['customer_id']==x]['customer_name'].values[0]}"
        )
        selected_app = df[df['customer_id'] == selected_id].iloc[0]
        
        # Application Summary Cards
        cols = st.columns(4)
        with cols[0]:
            st.metric("Customer", selected_app['customer_name'])
        with cols[1]:
            st.metric("Loan Amount", f"${selected_app['Loan Amount']:,.2f}")
        with cols[2]:
            st.metric("Loan Type", selected_app['Loan Type'])
        with cols[3]:
            st.metric("Current Status", selected_app['approval_status'])
        
        # Detailed View
        with st.expander("📄 Full Application Details", expanded=True):
            tab1, tab2 = st.tabs(["Basic Info", "Financial Metrics"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Application Date:**", selected_app['application_date'])
                    st.write("**Credit History:**", f"{selected_app['Credit History']}/5")
                    st.write("**Employment Status:**", selected_app.get('Employee_Status', 'N/A'))
                with col2:
                    st.write("**Location:**", selected_app.get('Location', 'N/A'))
                    st.write("**Collateral Type:**", selected_app.get('Collateral Type', 'N/A'))
                    st.write("**Collateral Value:**", f"${selected_app.get('Collateral Value', 0):,.2f}")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**DTI Ratio:**", f"{selected_app['Dti']:.1f}%")
                    st.write("**LVR Ratio:**", f"{selected_app['LVR']:.1f}%")
                with col2:
                    st.write("**Interest Rate:**", f"{selected_app.get('Interest Rate', 0):.1f}%")
                    st.write("**Loan Term:**", f"{selected_app.get('Loan Term', 0)} months")
        
        # Decision Panel
        st.markdown("### Final Decision")
        decision_cols = st.columns([1, 1, 2])
        
        with decision_cols[0]:
            if st.button("✅ Approve Application", 
                        use_container_width=True,
                        type="primary"):
                if update_decision(selected_id, "Approved"):
                    st.success("Application approved in database!")
                    st.rerun()
        
        with decision_cols[1]:
            if st.button("❌ Reject Application", 
                        use_container_width=True,
                        type="secondary"):
                if update_decision(selected_id, "Rejected"):
                    st.success("Application rejected in database!")
                    st.rerun()
        
        with decision_cols[2]:
            rejection_reason = st.text_area(
                "Rejection Notes (Required if rejecting)",
                placeholder="Enter detailed reason for rejection...",
                disabled=selected_app['final_decision'] != "Rejected"
            )
            
            if selected_app['final_decision'] == "Rejected" and not rejection_reason:
                st.warning("Please provide rejection reason before submitting")
def authenticate_user(username, password):
    """Validate credentials and set user role"""
    if username == "Risk" and password == "9988":
        st.session_state.user_role = "loan_officer"
        return True
    elif username == "Risk" and password == "9988":
        st.session_state.user_role = "manager"
        return True
    return False
# ----------- Main Page (Login + Summary) -----------
def show_home_page():
    # Top Row
    top_col1, top_col2 = st.columns([1, 4])
    with top_col1:
        image_path = os.path.join(BASE_DIR, "frontend", "Logo-CMCB.png")
        st.image(image_path, width=100)
    with top_col2:
        st.markdown("<h1 style='text-align: left; color: green;'>INTELLIGENT LOAN APPROVAL SYSTEM</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    
    # In your dashboard code:
    with col1:
        st.markdown("""
        <div class='table-container' style='text-align: center; font-weight: bold; font-size: 20px; color: #008A04;'>
            📊 Summary Loan Case This Month
        </div>
        """, unsafe_allow_html=True)

        branch_summary = get_branch_summary()
    
        if not branch_summary.empty:
        # Add numbering (after renaming columns)
            branch_summary.insert(0, 'No.', range(1, len(branch_summary)+1))
        
            st.dataframe(
                branch_summary,
                column_config={
                    "No.": st.column_config.NumberColumn(width="small"),
                    "Branch Code": st.column_config.TextColumn(width="medium"),
                    "Branch Name": st.column_config.TextColumn(width="large")
                },
                hide_index=True,
               use_container_width=True
             )
        else:
            st.warning("No branch data available for this month")

    with col2:
        st.markdown("<div class='header'>🔐 Login</div>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                login_button = st.form_submit_button("Login")

            if login_button:
                # Check login credentials
                if username == "RM" and password == "9876":
                    st.session_state.authenticated = True
                    st.session_state.username = "RM"
                    st.success("✅ Logged in as RM")
                    st.rerun()
                elif username == "Risk" and password == "9999":
                    st.session_state.authenticated = True
                    st.session_state.username = "Risk"
                    st.success("✅ Logged in as Manager")
                    st.rerun()
                elif username == "Board" and password == "9999":
                    st.session_state.authenticated = True
                    st.session_state.username = "Board"
                else:
                    st.error("❌ Invalid credentials")
def load_dice_person():
    # 1. Load model and data
    model_path = os.path.join(BASE_DIR, "frontend", "person_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "person_data.csv")

    hl_gb_model = joblib.load(model_path)
    data = pd.read_csv(data_path)

    # 2. Select and define features
    data = data[['Age', 'Income', 'Dti', 'Interest Rate', 'Credit History', 'Loan Amount',
                 'Guarantor Binary', 'Employee_Status', 'Customer Status', 'Loan Binary Status']]
    

    categorical_features = ['Employee_Status', 'Customer Status']
    numerical_features = ['Age', 'Income', 'Dti', 'Interest Rate', 'Credit History', 'Guarantor Binary', 'Loan Amount']
    #immutable_features = ['Application_ID']
    target = 'Loan Binary Status'

    # 3. Initialize DiCE
    dice_data = dice_ml.Data(
        dataframe=data,
        continuous_features=numerical_features,
        categorical_features=categorical_features,
        outcome_name=target,
    )

    dice_model = dice_ml.Model(model=hl_gb_model, backend="sklearn")
    explainer = Dice(dice_data, dice_model, method="random")

    return explainer, hl_gb_model, numerical_features + categorical_features
def load_dice_hl():
    # 1. Load model and data
    model_path = os.path.join(BASE_DIR, "frontend", "hl_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "hl_data.csv")
    #model_path = '/frontend/hl_pipeline.pkl'
    #data_path = '/hl_data.csv'

    hl_gb_model = joblib.load(model_path)
    data = pd.read_csv(data_path)

    # 2. Select and define features
    data = data[['Income', 'Dti', 'LVR', 'Interest Rate', 'Credit History', 
             'Guarantor Binary', 'Employee_Status', 'Loan Term', 
             'Collateral Type', 'Loan Binary Status']]

    # 2. Define features
    numerical_features = ['Income', 'Dti', 'LVR', 'Interest Rate', 'Credit History', 'Guarantor Binary']
    categorical_features = ['Employee_Status', 'Loan Term', 'Collateral Type']
    target = 'Loan Binary Status'

    # 3. Initialize DiCE
    dice_data = dice_ml.Data(
        dataframe=data,
        continuous_features=numerical_features,
        categorical_features=categorical_features,
        outcome_name=target,
    )

    dice_model = dice_ml.Model(model=hl_gb_model, backend="sklearn")
    explainer = Dice(dice_data, dice_model, method="random")

    return explainer, hl_gb_model, numerical_features + categorical_features
def load_dice_sme():
    # 1. Load model and data
    model_path = os.path.join(BASE_DIR, "frontend", "sme_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "sme_data.csv")
    #model_path = '/frontend/sme_pipeline.pkl'
    #data_path = '/sme_data.csv'

    hl_gb_model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    
    data = data[['Dti', 'Interest Rate', 'LVR' , 'Employee_Status', 'Loan Term', 'Credit History', 'Guarantor Binary', 'Loan Binary Status']]

    # 2. Define features
    numerical_features = ['Dti', 'Interest Rate', 'LVR', 'Credit History', 'Guarantor Binary']
    categorical_features = ['Employee_Status', 'Loan Term']
    target = 'Loan Binary Status'

    # 3. Initialize DiCE
    dice_data = dice_ml.Data(
        dataframe=data,
        continuous_features=numerical_features,
        categorical_features=categorical_features,
        outcome_name=target,
    )

    dice_model = dice_ml.Model(model=hl_gb_model, backend="sklearn")
    explainer = Dice(dice_data, dice_model, method="random")

    return explainer, hl_gb_model, numerical_features + categorical_features
def generate_llm_advice(changes: list, loan_type: str) -> str:
    prompt = f"""You are a financial advisor helping {loan_type} loan applicants improve their approval chances.
Convert these changes into business-friendly suggestions for sales teams to guide customers:
Changes: {"; ".join(changes)}

Provide 3-4 sentences explaining why these changes matter, how they help, and motivate the customer to improve income, creditworthiness, or collateral."""
    
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                'model': 'llama3.2',
                'messages': [
                    {'role': 'system', 'content': 'You are a financial advisor helping loan applicants understand and improve approval chances.'},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False
            }
        )
        return response.json()['message']['content']
    
    except Exception as e:
        return f"*AI Advice could not be generated due to: {str(e)}*"

def show_form_page():
    # Header Section
    top_col1, top_col2 = st.columns([1, 4])
    with top_col1:
        image_path = os.path.join(BASE_DIR, "frontend", "Logo-CMCB.png")
        st.image(image_path, width=100)
    with top_col2:
        st.markdown("<h1 style='text-align: left; color: green;'>Loan Intelligent Approval System</h1>",
                    unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #004A08;
                padding: 8px 15px;
                border-radius: 30px;
                margin: 10px 0px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: white; margin: 0; font-size: 1.2em;'>📝 Customer Information Form</h2>
    </div>
    """, unsafe_allow_html=True)
    # Move loan_type outside form to control loan_term dynamically
    loan_type = st.selectbox("Loan Type",
                             ["Personal Loan", "SME Loan", "Home Loan"],
                             key="loan_type_outside")
    # Define loan term options dynamically
    if loan_type == "Home Loan":
        loan_term_options = [120, 180, 240, 300, 360]
    elif loan_type == "SME Loan":
        loan_term_options = [36, 48, 60, 72, 84, 96, 108]
    else:
        loan_term_options = [12, 18, 24, 30, 36, 48]

    # Agree Checkbox
    agree = st.checkbox("I certify that all information provided is accurate and complete",
                        key="agree_outside_form")
    # Start Form
    with st.form("loan_application_form"):
        rm_code = st.selectbox("Relationship Manager Code (RM_Code)", 
                               [f"RM{str(i).zfill(2)}" for i in range(1, 21)],
                               key="rm_code")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            customer_name = st.text_input("Full Name", key="name")
            customer_age = st.number_input("Age", min_value=18, max_value=80, key="age")
            customer_location = st.selectbox("Location",
                                             ["Phnom Penh", "Siem Reap", "Battambang", "Other"],
                                             key="location")
            # re-use loan_type from outside
            st.markdown(f"**Loan Type:** `{loan_type}`")

        with col2:
            marital_status = st.selectbox("Marital Status",
                                          ["Single", "Married", "Divorced"],
                                          key="marital")
            collateral_type = st.selectbox("Collateral Type",
                                           ["Car", "House", "Land"],
                                           key="collateral")
            collateral_value = st.number_input("Collateral Value ($)",
                                               min_value=0.0,
                                               value=0.0,
                                               key="collateral_val")
            employee_status = st.selectbox("Employment Status",
                                           ["Full-time", "Part-time", "Self-employed", "Unemployed"],
                                           key="employment")

        with col3:
            income = st.number_input("Monthly Income ($)",
                                     min_value=0.0,
                                     key="income")
            loan_amount = st.number_input("Loan Amount ($)",
                                          min_value=0.0,
                                          key="loan_amt")
            loan_term = st.selectbox("Loan Term (Months)",
                                     loan_term_options,
                                     key="loan_term")
            debt = st.number_input("Monthly Debt ($)",
                                   min_value=0.0,
                                   value=0.0,
                                   key="debt")

        with col4:
            guarantor = st.selectbox("Has Guarantor?",
                                     ["Yes", "No"],
                                     key="guarantor")
            customer_status = st.selectbox("Customer Type",
                                           ["New Customer", "Existing Customer"],
                                           key="cust_status")
            customer_relation_bank = st.selectbox("Bank Products",
                                                  ['None', 'KHQR', 'Loan', 'Deposit', 'Card', 'Multiple'],
                                                  key="bank_relation")
            interest_rate = st.number_input("Interest Rate (%)",
                                            min_value=0.0,
                                            max_value=25.0,
                                            value=12.0,
                                            step=0.1,
                                            key="interest")
        st.markdown("""
        <div style='background-color: #f5f5f5; padding: 15px; border-radius: 10px; border: 1px solid #ccc; margin-top: 20px;'>
            <h4 style='color: #004A08;'>📎 Required Documents Upload</h4>
            <p style='color: #555;'>Please upload clear and valid copies of the following documents:</p>
        </div>
        """, unsafe_allow_html=True)

        # Side-by-side uploader
        doc_col1, doc_col2 = st.columns(2)

        with doc_col1:
            id_card = st.file_uploader("🪪 National ID Card", type=["pdf", "png", "jpg", "jpeg"], key="id_card")

        with doc_col2:
            land_plan = st.file_uploader("🗺️ Land Plan Document", type=["pdf", "png", "jpg", "jpeg"], key="land_plan")

        # Read bytes (for database)
        id_card_bytes = id_card.read() if id_card else None
        land_plan_bytes = land_plan.read() if land_plan else None
        
        
        loan_reason = st.text_area("Loan Purpose",
                                   help="Please describe the purpose of this loan",
                                   key="purpose")
        credit_history = st.select_slider("Credit Score",
                                          options=["1", "2", "3", "4", "5"],
                                          key="credit_score")

        submitted = st.form_submit_button("Submit Application",
                                          disabled=not agree,
                                          help="Please agree to the terms before submitting")

        if submitted:
            # Ensure income and loan_amount are not zero if they're critical
            if not all([customer_name, customer_age, customer_location, loan_type, id_card,
                        income is not None, loan_amount is not None, income > 0, loan_amount > 0]): # Added checks for income/loan_amount
                st.error("Please fill in all required fields (Name, Age, Location, Loan Type) and ensure Income and Loan Amount are greater than zero.")
            else:
                # Collect all form data
                form_data = {
                    'customer_name': customer_name,
                    'customer_age': customer_age,
                    'customer_location': customer_location,
                    'loan_type': loan_type,
                    'marital_status': marital_status,
                    'collateral_type': collateral_type,
                    'collateral_value': collateral_value,
                    'employee_status': employee_status,
                    'income': income,
                    'loan_term': loan_term,
                    'loan_amount': loan_amount,
                    'debt': debt,
                    'guarantor': guarantor,
                    'customer_status': customer_status,
                    'customer_relation_bank': customer_relation_bank,
                    'interest_rate': interest_rate,
                    'loan_reason': loan_reason,
                    'credit_history': credit_history,
                    'rm_code': rm_code,
                    'id_card': id_card_bytes,
                    'land_plan': land_plan_bytes
                }
                

                # Show processing animation
                with st.spinner('Processing your application...'):
    # Save to database first
                    if save_to_database(form_data):
                        st.success("Application submitted and saved to database successfully!")
                        st.balloons()
                        
                        # Show submission details
                        st.markdown("### Application Summary")
                        submission_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.write(f"**Application ID:** {st.session_state.last_application_id_generated}")
                        st.write(f"**Applicant Name:** {customer_name}")
                        st.write(f"**Loan Amount:** ${loan_amount:,.2f}")
                        st.write(f"**Loan Type:** {loan_type}")
                        st.write(f"**Submission Time:** {submission_time}")
                        
                        # Only predict for home loans
                        if form_data['loan_type'] in ['Home Loan', 'Personal Loan', 'SME Loan']:
                            prediction = predict_loan_approval(form_data, form_data['loan_type'])
                            if prediction:
                                st.markdown("### Model Prediction")
                                st.success(f"**Decision:** {prediction['prediction']}")
                                st.metric("Confidence Score", f"{prediction['confidence']:.2f}%")

                                # Detailed prediction analysis
                                with st.expander("📊 Prediction Analysis"):
                                    features = prediction['features']
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.markdown("**Key Factors**")
                                        if form_data['loan_type'] == 'Home Loan':
                                            st.write(f"• Income: ${features['Income']:,.2f}")
                                            st.write(f"• DTI (Debt-to-Income): {features['Dti']:.1f}%")
                                            st.write(f"• LVR (Loan-to-Value): {features['LVR']:.1f}%")
                                            st.write(f"• Interest Rate: {features['Interest Rate']}%")
                                            st.write(f"• Credit History: {features['Credit History']}")
                                            st.write(f"• Guarantor: {'Yes' if features['Guarantor Binary'] == 1 else 'No'}")
                                            st.write(f"• Employment: {features['Employee_Status']}")
                                            st.write(f"• Term: {features['Loan Term']} months")
                                            st.write(f"• Collateral: {features['Collateral Type']}")

                                        elif form_data['loan_type'] == 'SME Loan':
                                            st.write(f"• DTI (Debt-to-Income): {features['Dti']:.1f}%")
                                            st.write(f"• LVR (Loan-to-Value): {features['LVR']:.1f}%")
                                            st.write(f"• Interest Rate: {features['Interest Rate']}%")
                                            st.write(f"• Credit History: {features['Credit History']} years")
                                            st.write(f"• Guarantor: {'Yes' if features['Guarantor Binary'] == 1 else 'No'}")
                                            st.write(f"• Employment: {features['Employee_Status']}")
                                            st.write(f"• Term: {features['Loan Term']} months")

                                        elif form_data['loan_type'] == 'Personal Loan':
                                            st.write(f"• Age: {features['Age']} years")
                                            st.write(f"• Income: ${features['Income']:,.2f}")
                                            st.write(f"• DTI (Debt-to-Income): {features['Dti']:.1f}%")
                                            st.write(f"• Interest Rate: {features['Interest Rate']}%")
                                            st.write(f"• Loan Amount: ${features['Loan Amount']}")
                                            st.write(f"• Credit History: {features['Credit History']} years")
                                            st.write(f"• Guarantor: {'Yes' if features['Guarantor Binary'] == 1 else 'No'}")
                                            st.write(f"• Employment: {features['Employee_Status']}")
                                            st.write(f"• Customer Status: {features['Customer Status']}")
                                        
                                    with col2:
                                        try:
                                            import shap
                                            import matplotlib.pyplot as plt

                                            st.markdown("**Feature Impact (SHAP)**")

                                            # Use same key as model loading
                                            model_key = f"{form_data['loan_type'].lower().replace(' ', '_')}_model"
                                            pipeline = st.session_state[model_key]

                                            # Prepare input data (raw)
                                            raw_input_df = pd.DataFrame([prediction['features']])

                                            # Preprocess input using pipeline (excluding classifier)
                                            preprocessor = pipeline.named_steps.get('preprocessor')  # or however you named it
                                            if preprocessor is None:
                                                raise ValueError("Pipeline does not contain a 'preprocessor' step.")

                                            processed_input = preprocessor.transform(raw_input_df)
                                            
                                            
                                            # ✅ Get feature names after transformation
                                            feature_names = preprocessor.get_feature_names_out()
                                            processed_df = pd.DataFrame(processed_input, columns=feature_names)

                                            # Extract classifier
                                            classifier = pipeline.named_steps.get('classifier')
                                            if classifier is None:
                                                raise ValueError("Model pipeline does not contain a 'classifier' step.")

                                            # SHAP explanation
                                            explainer = shap.TreeExplainer(classifier)
                                            shap_values = explainer.shap_values(processed_df)

                                            if isinstance(shap_values, list):  # Binary classification
                                                shap_values = shap_values[1]

                                            # SHAP summary bar plot
                                            plt.figure()
                                            shap.summary_plot(
                                                shap_values,
                                                processed_df,
                                                plot_type="bar",
                                                show=False,
                                                plot_size=(8, 6)
                                            )
                                            st.pyplot(plt.gcf(), clear_figure=True)
                                            plt.close()
                                        except Exception as e:
                                            st.warning("Feature explanation could not be generated.")
                                            st.info(f"Reason: {str(e)}")
                                with st.expander("📊 Suggestions to Improve Loan Approval Chances"):           
                                # Show Counterfactual Suggestions if Rejected
                                    if prediction['prediction'] == 'Rejected':
                                                try:
                                                    st.markdown("**Suggestions to Get Loan Approved (Counterfactuals)**")
                                                    if loan_type == 'Personal Loan':
                                                        explainer, dice_model, dice_features = load_dice_person()

                                                        # Prepare input for DiCE (must match feature names)
                                                        input_for_dice = {k: prediction['features'].get(k, None) for k in dice_features}

                                                        # Create query instance
                                                        query_instance = pd.DataFrame([input_for_dice])

                                                        # Generate counterfactuals
                                                        dice_exp = explainer.generate_counterfactuals(
                                                            query_instances=query_instance,
                                                            total_CFs=3,
                                                            desired_class="opposite"
                                                        )

                                                        # Retrieve counterfactual dataframe
                                                        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

                                                        # Add Application_ID
                                                        cf_df['Application_ID'] = st.session_state.last_application_id_generated

                                                        # Reorder columns
                                                        cf_df = cf_df[['Application_ID'] + [col for col in cf_df.columns if col != 'Application_ID']]

                                                        # ----------------------------
                                                        # Generate human-readable suggestions
                                                        # ----------------------------
                                                        original = query_instance.iloc[0]
                                                        suggestions = []

                                                        for idx, cf in cf_df.iterrows():
                                                            # Prepare the differences for LLM
                                                            changes = []
                                                            for col in query_instance.columns:
                                                                original_val = original[col]
                                                                cf_val = cf[col]
                                                                
                                                                if original_val != cf_val:
                                                                    if isinstance(original_val, (int, float)):
                                                                        delta = round(cf_val - original_val, 2)
                                                                        direction = "increase" if delta > 0 else "decrease"
                                                                        changes.append(f"{col}: {direction} by {abs(delta)}")
                                                                    else:
                                                                        changes.append(f"{col}: change from '{original_val}' to '{cf_val}'")

                                                            # Query Llama3.2 for business-friendly explanation
                                                            try:
                                                                llm_advice = generate_llm_advice(changes, loan_type)
                                                                suggestions.append(f"**Option {idx+1}**\n{llm_advice}\n\n*Technical changes:* {', '.join(changes)}")
                                                            except Exception as e:
                                                                # Fallback to original format if LLM fails
                                                                changes = []
                                                                for col in query_instance.columns:
                                                                    original_val = original[col]
                                                                    cf_val = cf[col]
                                                                    if original_val != cf_val:
                                                                        if isinstance(original_val, (int, float)):
                                                                            delta = round(cf_val - original_val, 2)
                                                                            direction = "increase" if delta > 0 else "decrease"
                                                                            changes.append(f"{direction.capitalize()} {col} by {abs(delta)}")
                                                                        else:
                                                                            changes.append(f"Change {col} from '{original_val}' to '{cf_val}'")
                                                                suggestions.append("👉 " + "; ".join(changes))

                                                        # ----------------------------
                                                        # Show results in Streamlit
                                                        # ----------------------------
                                                        st.markdown("### 📊 Counterfactual Suggestions for SME Loan")
                                                        st.dataframe(cf_df)

                                                        st.markdown("### 💡 AI-Enhanced Business Advice")
                                                        for suggestion in suggestions:
                                                            st.markdown(suggestion)
                                                            st.divider()


                                                    elif loan_type == 'Home Loan':
                                                        explainer, dice_model, dice_features = load_dice_hl()

                                                        # Prepare input for DiCE (must match feature names)
                                                        input_for_dice = {k: prediction['features'].get(k, None) for k in dice_features}

                                                        # Create query instance
                                                        query_instance = pd.DataFrame([input_for_dice])

                                                        # Generate counterfactuals
                                                        dice_exp = explainer.generate_counterfactuals(
                                                            query_instances=query_instance,
                                                            total_CFs=3,
                                                            desired_class="opposite"
                                                        )

                                                        # Retrieve counterfactual dataframe
                                                        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

                                                        # Add Application_ID
                                                        cf_df['Application_ID'] = st.session_state.last_application_id_generated

                                                        # Reorder columns
                                                        cf_df = cf_df[['Application_ID'] + [col for col in cf_df.columns if col != 'Application_ID']]

                                                        # ----------------------------
                                                        # Generate human-readable suggestions
                                                        # ----------------------------
                                                        original = query_instance.iloc[0]
                                                        suggestions = []
                                                        for idx, cf in cf_df.iterrows():
                                                            # Prepare the differences for LLM
                                                            changes = []
                                                            for col in query_instance.columns:
                                                                original_val = original[col]
                                                                cf_val = cf[col]
                                                                
                                                                if original_val != cf_val:
                                                                    if isinstance(original_val, (int, float)):
                                                                        delta = round(cf_val - original_val, 2)
                                                                        direction = "increase" if delta > 0 else "decrease"
                                                                        changes.append(f"{col}: {direction} by {abs(delta)}")
                                                                    else:
                                                                        changes.append(f"{col}: change from '{original_val}' to '{cf_val}'")

                                                            # Query Llama3.2 for business-friendly explanation
                                                            try:
                                                                llm_advice = generate_llm_advice(changes, loan_type)
                                                                suggestions.append(f"**Option {idx+1}**\n{llm_advice}\n\n*Technical changes:* {', '.join(changes)}")
                                                            except Exception as e:
                                                                # Fallback to original format if LLM fails
                                                                changes = []
                                                                for col in query_instance.columns:
                                                                    original_val = original[col]
                                                                    cf_val = cf[col]
                                                                    if original_val != cf_val:
                                                                        if isinstance(original_val, (int, float)):
                                                                            delta = round(cf_val - original_val, 2)
                                                                            direction = "increase" if delta > 0 else "decrease"
                                                                            changes.append(f"{direction.capitalize()} {col} by {abs(delta)}")
                                                                        else:
                                                                            changes.append(f"Change {col} from '{original_val}' to '{cf_val}'")
                                                                suggestions.append("👉 " + "; ".join(changes))

                                                        # ----------------------------
                                                        # Show results in Streamlit
                                                        # ----------------------------
                                                        st.markdown("### 📊 Counterfactual Suggestions for SME Loan")
                                                        st.dataframe(cf_df)

                                                        st.markdown("### 💡 AI-Enhanced Business Advice")
                                                        for suggestion in suggestions:
                                                            st.markdown(suggestion)
                                                            st.divider()
                                                    else:
                                                        explainer, dice_model, dice_features = load_dice_sme()

                                                        # Prepare input for DiCE (must match feature names)
                                                        input_for_dice = {k: prediction['features'].get(k, None) for k in dice_features}

                                                        # Create query instance
                                                        query_instance = pd.DataFrame([input_for_dice])

                                                        # Generate counterfactuals
                                                        dice_exp = explainer.generate_counterfactuals(
                                                            query_instances=query_instance,
                                                            total_CFs=3,
                                                            desired_class="opposite"
                                                        )

                                                        # Retrieve counterfactual dataframe
                                                        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

                                                        # Add Application_ID
                                                        cf_df['Application_ID'] = st.session_state.last_application_id_generated

                                                        # Reorder columns
                                                        cf_df = cf_df[['Application_ID'] + [col for col in cf_df.columns if col != 'Application_ID']]

                                                        # ----------------------------
                                                        # Generate human-readable suggestions using Llama3.2
                                                        # ----------------------------
                                                        original = query_instance.iloc[0]
                                                        suggestions = []

                                                        for idx, cf in cf_df.iterrows():
                                                            # Prepare the differences for LLM
                                                            changes = []
                                                            for col in query_instance.columns:
                                                                original_val = original[col]
                                                                cf_val = cf[col]
                                                                
                                                                if original_val != cf_val:
                                                                    if isinstance(original_val, (int, float)):
                                                                        delta = round(cf_val - original_val, 2)
                                                                        direction = "increase" if delta > 0 else "decrease"
                                                                        changes.append(f"{col}: {direction} by {abs(delta)}")
                                                                    else:
                                                                        changes.append(f"{col}: change from '{original_val}' to '{cf_val}'")

                                                            # Query Llama3.2 for business-friendly explanation
                                                            try:
                                                                llm_advice = generate_llm_advice(changes, loan_type)
                                                                suggestions.append(f"**Option {idx+1}**\n{llm_advice}\n\n*Technical changes:* {', '.join(changes)}")
                                                            except Exception as e:
                                                                # Fallback to original format if LLM fails
                                                                changes = []
                                                                for col in query_instance.columns:
                                                                    original_val = original[col]
                                                                    cf_val = cf[col]
                                                                    if original_val != cf_val:
                                                                        if isinstance(original_val, (int, float)):
                                                                            delta = round(cf_val - original_val, 2)
                                                                            direction = "increase" if delta > 0 else "decrease"
                                                                            changes.append(f"{direction.capitalize()} {col} by {abs(delta)}")
                                                                        else:
                                                                            changes.append(f"Change {col} from '{original_val}' to '{cf_val}'")
                                                                suggestions.append("👉 " + "; ".join(changes))
                                                        # ----------------------------
                                                        # Show results in Streamlit
                                                        # ----------------------------
                                                        st.markdown("### 📊 Counterfactual Suggestions for SME Loan")
                                                        st.dataframe(cf_df)

                                                        st.markdown("### 💡 AI-Enhanced Business Advice")
                                                        for suggestion in suggestions:
                                                            st.markdown(suggestion)
                                                            st.divider()

                                                        st.markdown("✅ Try adjusting the values above to improve loan approval chances.")
                                                except requests.exceptions.RequestException as e:
                                                    st.error(f"Failed to connect to AI model: {str(e)}")
                                                    st.info("Showing basic suggestions without AI enhancement")
                                                    
                                                except Exception as e:
                                                    st.error(f"An unexpected error occurred: {str(e)}")
                                                    st.info("Please try again or contact support")
def get_branch_summary():
    """Fetch summary statistics by branch using RM_Code mapping"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    b.branch_code,
                    b.branch_name,
                    COUNT(l."Application_ID") as total_cases,
                    SUM(CASE WHEN l."Loan Status" = 'Pending' THEN 1 ELSE 0 END) as case_pending,
                    SUM(CASE WHEN l."Loan Status" = 'Rejected' THEN 1 ELSE 0 END) as cases_rejected,
                    SUM(CASE WHEN l."Loan Status" = 'Approved' THEN 1 ELSE 0 END) as cases_approved
                FROM branch_reference b
                LEFT JOIN loancamdata l ON l."RM_Code" = ANY(b.rm_codes)
                WHERE TO_DATE(l."Date", 'YYYY-MM-DD') >= DATE_TRUNC('month', CURRENT_DATE)
                GROUP BY b.branch_code, b.branch_name
                ORDER BY b.branch_code
            """)
            
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            # Rename columns for display
            column_mapping = {
                'branch_code': 'Branch Code',
                'branch_name': 'Branch Name',
                'total_cases': 'Total Application',
                'case_pending': 'Pending Applicant',
                'cases_rejected': 'Rejected Applicant',
                'cases_approved': 'Approved Applicant'
            }
            df = df.rename(columns=column_mapping)
            
            return df
            
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()

def load_loan_data():
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT * FROM loancamdata
            """)
            result = conn.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except SQLAlchemyError as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    return df

import plotly.express as px
import geopandas as gpd

def show_stakeholder_dashboard():
    # =============== HEADER SECTION ===============
    top_col1, top_col2 = st.columns([1, 4])
    with top_col1:
        image_path = os.path.join(BASE_DIR, "frontend", "Logo-CMCB.png")
        st.image(image_path, width=120)  # update path accordingly
    with top_col2:
        st.markdown(
            "<h1 style='text-align: left; color: #0a7d34;'>📊 Stakeholder Dashboard</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("### Loan Portfolio Insights & Risk Overview")

    df = load_loan_data()
    df['Date'] = pd.to_datetime(df['Date'])

    # =============== KPI CARDS ===============
    total_loans = len(df)
    total_approved = df[df['Loan Status'] == 'Approved']['Loan Amount'].sum()
    total_rejected = df[df['Loan Status'] == 'Rejected']['Loan Amount'].sum()
    approval_rate = (df['Loan Status'] == 'Approved').mean() * 100

    st.markdown("### 🔑 Key Performance Indicators")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("📁 Total Loan Cases", f"{total_loans:,}")
    kpi2.metric("✅ Approval Rate", f"{approval_rate:.2f}%")
    kpi3.metric("💰 Approved Amount", f"${total_approved:,.0f}")

    st.divider()

        # =============== MAP (PLOTLY) ===============
    st.subheader("🌍 Loan Disbursements by Province")

    geojson_path = os.path.join(BASE_DIR, "frontend", "geoBoundaries-KHM-ADM1_simplified.geojson")
    cambodia = gpd.read_file(geojson_path)
    disbursements = pd.DataFrame({
        'Province': ['Stung Treng', 'Phnom Penh', 'Kampong Thom', 'Koh Kong'],
        'Amount': [500000, 1200000, 750000, 300000]
    })

    merged = cambodia.merge(disbursements,
                            left_on='shapeName',
                            right_on='Province',
                            how='left')

    # 2. Plot with matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    merged.plot(column='Amount',
                ax=ax,
                legend=False,
                cmap='YlOrRd',
                missing_kwds={'color': 'lightgrey'},
                edgecolor='black')
    plt.title("CMCB Disbursements by Province")
    plt.axis('off')
    for idx, row in merged.iterrows():
        plt.annotate(text=f"{row['shapeName']}\n${row['Amount']:,}",
                    xy=row['geometry'].centroid.coords[0],
                    ha='center',
                    fontsize=4)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.pyplot(fig, clear_figure=True)

    with col2:
        # Calculate metrics
        total_disbursed = disbursements['Amount'].sum()
        avg_per_province = disbursements['Amount'].mean()
        max_province = disbursements.loc[disbursements['Amount'].idxmax()]
        min_province = disbursements.loc[disbursements['Amount'].idxmin()]
        
        # Create a beautiful metrics dashboard
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0a7d34 0%, #0d9742 100%); 
                    padding: 20px; 
                    border-radius: 15px; 
                    color: white;
                    margin-bottom: 20px;'>
            <h3 style='color: white; margin: 0; text-align: center;'>📊 Disbursement Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Cards
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="💰 Total Disbursed",
                value=f"${total_disbursed:,.0f}",
                help="Total loan amount across all provinces"
            )
            
            st.metric(
                label="📈 Average per Province",
                value=f"${avg_per_province:,.0f}",
                help="Average loan amount per province"
            )
        
        with metric_col2:
            st.metric(
                label="🏆 Highest Province",
                value=f"{max_province['Province']}",
                delta=f"${max_province['Amount']:,.0f}",
                help="Province with highest disbursement"
            )
            
            st.metric(
                label="📉 Lowest Province",
                value=f"{min_province['Province']}",
                delta=f"${min_province['Amount']:,.0f}",
                help="Province with lowest disbursement"
            )
        
        # Progress bars for each province
        # Quick insights
        #st.markdown("### 💡 Key Observations")
        
        #insights = [
        #    f"• **Phnom Penh leads** with ${max_province['Amount']:,.0f} ({(max_province['Amount']/total_disbursed)*100:.1f}% of total)",
        #    f"• **{min_province['Province']} has the lowest** disbursement at ${min_province['Amount']:,.0f}",
        #    f"• **Average disbursement** per province: ${avg_per_province:,.0f}",
        #    f"• **{len(disbursements)} provinces** currently active in lending program"
        #]
        #
        #for insight in insights:
        #    st.markdown(f"""
        #    <div style='background: #e8f5e8; padding: 8px 12px; margin: 5px 0; border-radius: 8px; border-left: 4px solid #0a7d34;'>
        #        {insight}
        #    </div>
        #    """, unsafe_allow_html=True)
    
    
    # Convert to GeoJSON for Plotly
    #merged_json = merged.to_json()

    #fig_map = px.choropleth_mapbox(
    #    merged,
    #    geojson=merged_json,
    #    locations="Province",
    #    color="Amount",
    #    featureidkey="properties.shapeName",
    #    hover_data={"shapeName": True, "Amount": True},
    #    hover_name="Province",
    #    color_continuous_scale="YlGn",
    #    mapbox_style="carto-positron",
    #    center={"lat": 12.5657, "lon": 104.9910},  # Cambodia center
    #    zoom=5,
    #    opacity=0.7,
    #    title="CMCB Disbursements by Province"
    #)
    #fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    #st.plotly_chart(fig_map, use_container_width=True)

    # =============== MONTHLY TREND ===============
    st.subheader("📅 Monthly Loan Amount Trend")
    monthly = df.groupby(df['Date'].dt.to_period('M')).agg({'Loan Amount': 'sum'}).reset_index()
    monthly['Date'] = monthly['Date'].astype(str)

    fig_trend = px.line(
        monthly,
        x='Date',
        y='Loan Amount',
        markers=True,
        title="📈 Total Loan Disbursed Per Month"
    )
    fig_trend.update_traces(line=dict(color='green', width=3))
    fig_trend.update_layout(template="plotly_white", xaxis_title="Month", yaxis_title="Loan Amount (USD)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # =============== RISK METRICS ===============
    #st.subheader("⚠️ Risk Profile Distributions")
    #col1, col2 = st.columns(2)
    #with col1:
    #    fig_dti = px.histogram(df, x='Dti', nbins=20, title='Debt-to-Income Ratio (DTI)')
    #    fig_dti.update_layout(template="plotly_white")
    #    st.plotly_chart(fig_dti, use_container_width=True)
    #with col2:
    #    fig_lvr = px.histogram(df, x='LVR', nbins=20, title='Loan-to-Value Ratio (LVR)')
    #    fig_lvr.update_layout(template="plotly_white")
    #    st.plotly_chart(fig_lvr, use_container_width=True)

    # =============== CUSTOMER PROFILE ===============
    st.subheader("👥 Customer Demographics")
    col1, col2 = st.columns(2)
    with col1:
        customer_df = df['Customer Status'].value_counts().reset_index()
        customer_df.columns = ['Status', 'Count']
        fig_customer = px.pie(customer_df, values='Count', names='Status', title='Customer Status')
        st.plotly_chart(fig_customer, use_container_width=True)
    with col2:
        employee_df = df['Employee_Status'].value_counts().reset_index()
        employee_df.columns = ['Employment', 'Count']
        fig_employee = px.pie(employee_df, values='Count', names='Employment', title='Employment Status')
        st.plotly_chart(fig_employee, use_container_width=True)
        
# -------- Main App Flow --------
if st.session_state.authenticated:
    if st.session_state.username == "RM":
        show_form_page()
    elif st.session_state.username == "Risk":
        show_approval_dashboard()
    elif st.session_state.username == "Board":
        show_stakeholder_dashboard()
else:
    show_home_page()
    
