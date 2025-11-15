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
        df = pd.read_csv('data/data.csv')
        return df
    except:
        st.error("Error loading data/data.csv. Please ensure the file exists in the data folder.")
        return None

# Loading models
@st.cache_resource
def load_models():
    "Load trained models"
    try:
        with open('../notebook/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('../notebook/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('../notebook/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, scaler, le
    except:
        return None, None, None

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
# 

# ============================================
# PAGE 2: DATA EXPLORATION
# ============================================
elif page == "Data Exploration":
    st.title("Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistics", "Correlations", 
                                       "Distributions", "Geographic"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Data Types & Missing Values")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            st.dataframe(pd.DataFrame(df.dtypes, columns=['Type']))
        
        with col2:
            st.write("**Missing Values:**")
            missing = df.isnull().sum()
            st.dataframe(pd.DataFrame(missing, columns=['Count']))
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        text_auto='.2f')
        fig.update_layout(height=600, title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Key Insights:**
        - Strong positive correlation between USD_CAP and TOTAL_SPEND
        - PC_GDP shows moderate correlation with other expenditure metrics
        - TIME shows temporal trends in healthcare spending
        """)
    
    with tab3:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", numeric_cols)
            
            fig = px.histogram(df, x=feature, nbins=50,
                             title=f"Distribution of {feature}",
                             labels={feature: feature},
                             color_discrete_sequence=['#1f77b4'])
            fig.add_vline(x=df[feature].mean(), line_dash="dash", 
                         line_color="red", annotation_text="Mean")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=feature, 
                        title=f"Box Plot - {feature}",
                        color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Country-wise Analysis")
        
        metric = st.selectbox("Select Metric", 
                             ['PC_GDP', 'USD_CAP', 'PC_HEALTHXP', 'TOTAL_SPEND'])
        
        country_avg = df.groupby('LOCATION')[metric].mean().sort_values(ascending=False).head(15)
        
        fig = px.bar(x=country_avg.index, y=country_avg.values,
                    labels={'x': 'Country', 'y': metric},
                    title=f"Top 15 Countries by Average {metric}",
                    color=country_avg.values,
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series by country
        st.subheader("Time Series Analysis")
        countries = st.multiselect("Select Countries", 
                                   df['LOCATION'].unique().tolist(),
                                   default=df['LOCATION'].unique().tolist()[:5])
        
        if countries:
            filtered_df = df[df['LOCATION'].isin(countries)]
            fig = px.line(filtered_df, x='TIME', y=metric, color='LOCATION',
                         title=f"{metric} Over Time by Country")
            st.plotly_chart(fig, use_container_width=True)