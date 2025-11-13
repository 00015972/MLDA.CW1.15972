"Streamlit app"
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Healthcare Expenditure ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Loading data
@st.cache_data
def load_data():
    "Loading the healthcare dataset"
    try:
        df = pd.read_csv('../data/data.csv')
        return df
    except:
        st.error("Error loading data/data.csv. Please ensure the file exists in the data folder.")
        return None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Home", "Data Exploration", "Data Preprocessing", 
     "Model Training", "Predictions", "Evaluation"]
)

# Load data
df = load_data()

if df is None:
    st.stop()

# ============================================
# PAGE 1: HOME
# ============================================
if page == "Home":
    st.markdown('<h1 class="main-header">Healthcare Expenditure Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Healthcare ML Dashboard
    
    This interactive application analyzes global healthcare expenditure patterns and provides 
    predictive insights for healthcare spending.
    
    ### Business Objective
    - **Predict healthcare expenditure** per capita based on historical trends
    - **Identify patterns** across countries and time periods
    - **Support policy decisions** for healthcare budget planning
    
    ### Dataset Overview
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Countries", df['LOCATION'].nunique())
    with col3:
        st.metric("Year Range", f"{df['TIME'].min()}-{df['TIME'].max()}")
    with col4:
        st.metric("Features", len(df.columns))
    
    st.markdown("### Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
