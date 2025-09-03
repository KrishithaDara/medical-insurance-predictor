import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1rem 0;
    font-weight: bold;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #2196f3;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.warning-box {
    background-color: #fff3e0;
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #ff9800;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success-box {
    background-color: #e8f5e8;
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #4caf50;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.sidebar-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Load and process data from CSV
@st.cache_data
def load_insurance_data():
    """Load and preprocess the insurance dataset from CSV"""
    try:
        # Try to load the CSV file
        data = pd.read_csv('med-insurance.csv')
        
        # Display basic info about the loaded data
        st.sidebar.success(f"✅ Data loaded: {len(data)} records")
        
        # Basic data cleaning
        data = data.dropna()  # Remove any missing values
        
        # Ensure proper column names (handle different naming conventions)
        expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        
        # Check if we have 'charges' or 'expenses' column
        if 'charges' in data.columns and 'expenses' not in data.columns:
            data = data.rename(columns={'charges': 'expenses'})
        elif 'expenses' not in data.columns and 'charges' not in data.columns:
            st.error("❌ CSV must contain a 'charges' or 'expenses' column")
            return None
            
        return data
        
    except FileNotFoundError:
        st.error("❌ File 'med-insurance.csv' not found. Please upload the file to the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# Preprocess data for machine learning
@st.cache_data
def preprocess_data(data):
    """Preprocess the insurance data for ML models"""
    if data is None:
        return None, None, None
    
    # Create a copy for processing
    processed_data = data.copy()
    
    # Initialize label encoders
    label_encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['sex', 'smoker', 'region']
    for col in categorical_columns:
        if col in processed_data.columns:
            le = LabelEncoder()
            processed_data[col + '_encoded'] = le.fit_transform(processed_data[col])
            label_encoders[col] = le
    
    # Create feature columns for ML
    feature_columns = ['age', 'bmi', 'children']
    for col in categorical_columns:
        if col in processed_data.columns:
            feature_columns.append(col + '_encoded')
    
    # Prepare features and target
    X = processed_data[feature_columns]
    y = processed_data['expenses']
    
    return X, y, label_encoders

# Train multiple ML models
@st.cache_resource
def train_models(X, y):
    """Train multiple ML models and return results"""
    if X is None or y is None:
        return None, None
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    results = {}
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    models['Linear Regression'] = lr_model
    results['Linear Regression'] = {
        'r2': r2_score(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'predictions': lr_pred
    }
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'r2': r2_score(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'predictions': rf_pred
    }
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'r2': r2_score(y_test, gb_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'predictions': gb_pred
    }
    
    # Ensemble (weighted average)
    ensemble_pred = 0.3 * lr_pred + 0.3 * rf_pred + 0.4 * gb_pred
    results['Ensemble'] = {
        'r2': r2_score(y_test, ensemble_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'predictions': ensemble_pred
    }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    
    return {
        'models': models,
        'results': results,
        'scaler': scaler,
        'best_model': best_model_name,
        'test_data': (X_test, y_test)
    }, scaler

# BMI category function
def get_bmi_category(bmi):
    """Get BMI category and color"""
    if bmi < 18.5:
        return "Underweight", "blue"
    elif 18.5 <= bmi < 25:
        return "Normal weight", "green"
    elif 25 <= bmi < 30:
        return "Overweight", "orange"
    else:
        return "Obese", "red"

# Main app
def main():
    # Title and Introduction
    st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What does this app do?</h3>
    <p>This AI-powered application predicts your annual medical insurance costs based on your personal health profile. 
    It uses advanced machine learning algorithms trained on real insurance data to provide accurate cost estimates.</p>
    <p><strong>📊 Data Source:</strong> Real medical insurance dataset with comprehensive health and demographic factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('📊 Loading insurance data...'):
        data = load_insurance_data()
    
    if data is None:
        st.stop()
    
    # Preprocess data
    with st.spinner('🔧 Preprocessing data...'):
        X, y, label_encoders = preprocess_data(data)
    
    if X is None:
        st.stop()
    
    # Train models
    with st.spinner('🤖 Training AI models...'):
        model_info, scaler = train_models(X, y)
    
    if model_info is None:
        st.stop()
    
    # Sidebar for user inputs
    st.sidebar.markdown('<h2 style="color: #1f77b4;">📝 Enter Your Information</h2>', unsafe_allow_html=True)
    
    # Personal Details Section
    st.sidebar.markdown("### 👤 Personal Details")
    age = st.sidebar.slider("🎂 Age", 18, 65, 30, help="Your current age in years")
    sex = st.sidebar.selectbox("👤 Gender", options=data['sex'].unique(), help="Your gender")
    
    # Health Information Section
    st.sidebar.markdown("### 🏥 Health Information")
    bmi = st.sidebar.slider("⚖️ BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1, 
                           help="BMI = weight(kg) / height(m)²")
    
    # BMI interpretation
    bmi_category, bmi_color = get_bmi_category(bmi)
    st.sidebar.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")
    
    smoker = st.sidebar.selectbox("🚭 Smoking Status", options=data['smoker'].unique(), 
                                 help="Do you currently smoke?")
    
    # Family & Location Section
    st.sidebar.markdown("### 👨‍👩‍👧‍👦 Family & Location")
    children = st.sidebar.selectbox("👶 Number of Children", options=sorted(data['children'].unique()),
                                   help="Number of dependent children covered by insurance")
    region = st.sidebar.selectbox("🌎 Region", options=data['region'].unique(),
                                 help="Your geographical region")
    
    # Display data summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="sidebar-info">
    <h4>📊 Dataset Summary</h4>
    <p><strong>Total Records:</strong> {len(data):,}</p>
    <p><strong>Average Cost:</strong> ${data['expenses'].mean():,.0f}</p>
    <p><strong>Age Range:</strong> {data['age'].min()}-{data['age'].max()} years</p>
    <p><strong>BMI Range:</strong> {data['bmi'].min():.1f}-{data['bmi'].max():.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Make prediction
    try:
        # Encode categorical inputs
        sex_encoded = label_encoders['sex'].transform([sex])[0]
        smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
        region_encoded = label_encoders['region'].transform([region])[0]
        
        # Prepare input for prediction
        user_input = np.array([[age, bmi, children, sex_encoded, smoker_encoded, region_encoded]])
        user_input_scaled = scaler.transform(user_input)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in model_info['models'].items():
            pred = model.predict(user_input_scaled)[0]
            predictions[model_name] = pred
        
        # Ensemble prediction
        ensemble_pred = (0.3 * predictions['Linear Regression'] + 
                        0.3 * predictions['Random Forest'] + 
                        0.4 * predictions['Gradient Boosting'])
        predictions['Ensemble'] = ensemble_pred
        
        # Main prediction display
        st.markdown('<h2 class="sub-header">🎯 Your Insurance Cost Prediction</h2>', unsafe_allow_html=True)
        
        # Best model prediction
        best_model = model_info['best_model']
        if best_model != 'Ensemble':
            final_prediction = predictions[best_model]
        else:
            final_prediction = ensemble_pred
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Predicted Annual Cost", 
                value=f"${final_prediction:,.0f}",
                help=f"Based on {best_model} model"
            )
        
        with col2:
            avg_cost = data['expenses'].mean()
            diff = final_prediction - avg_cost
            st.metric(
                label="📊 vs Dataset Average", 
                value=f"${avg_cost:,.0f}", 
                delta=f"{diff:+,.0f}",
                help="Comparison with dataset average"
            )
        
        with col3:
            monthly_cost = final_prediction / 12
            st.metric(
                label="📅 Monthly Premium", 
                value=f"${monthly_cost:,.0f}",
                help="Estimated monthly cost"
            )
        
        with col4:
            percentile = (data['expenses'] < final_prediction).mean() * 100
            st.metric(
                label="📈 Cost Percentile", 
                value=f"{percentile:.0f}%",
                help="Your cost compared to others in dataset"
            )
        
        # Model performance section
        st.markdown('<h2 class="sub-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
        
        perf_col1, perf_col2 = st.columns([1, 1])
        
        with perf_col1:
            # Model comparison
            model_names = list(model_info['results'].keys())
            r2_scores = [model_info['results'][name]['r2'] for name in model_names]
            
            fig = px.bar(
                x=model_names, 
                y=r2_scores,
                title="🔍 Model Accuracy Comparison (R² Score)",
                labels={'x': 'Model', 'y': 'R² Score'},
                color=r2_scores,
                color_continuous_scale='viridis',
                text=[f"{score:.3f}" for score in r2_scores]
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with perf_col2:
            # Best model info
            best_r2 = model_info['results'][best_model]['r2']
            best_rmse = model_info['results'][best_model]['rmse']
            
            st.markdown(f"""
            <div class="success-box">
            <h4>🏆 Best Performing Model</h4>
            <p><strong>Model:</strong> {best_model}</p>
            <p><strong>Accuracy (R²):</strong> {best_r2:.3f}</p>
            <p><strong>RMSE:</strong> ${best_rmse:,.0f}</p>
            <p><strong>Interpretation:</strong> {best_r2*100:.1f}% of cost variation explained</p>
            </div>
            """, unsafe_allow_html=True)
            
            # All model predictions
            st.markdown("#### 🔮 All Model Predictions")
            for model_name, pred in predictions.items():
                st.write(f"**{model_name}:** ${pred:,.0f}")
        
        # Cost factor analysis
        st.markdown('<h2 class="sub-header">📈 Cost Factor Analysis</h2>', unsafe_allow_html=True)
        
        # Interactive analysis
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "🎂 Age Impact", "⚖️ BMI Impact", "🚭 Smoking Impact", "🌎 Regional Comparison"
        ])
        
        with analysis_tab1:
            # Age impact analysis
            age_range = range(18, 66, 2)
            age_costs = []
            
            for test_age in age_range:
                test_input = np.array([[test_age, bmi, children, sex_encoded, smoker_encoded, region_encoded]])
                test_input_scaled = scaler.transform(test_input)
                if best_model != 'Ensemble':
                    cost = model_info['models'][best_model].predict(test_input_scaled)[0]
                else:
                    lr_pred = model_info['models']['Linear Regression'].predict(test_input_scaled)[0]
                    rf_pred = model_info['models']['Random Forest'].predict(test_input_scaled)[0]
                    gb_pred = model_info['models']['Gradient Boosting'].predict(test_input_scaled)[0]
                    cost = 0.3 * lr_pred + 0.3 * rf_pred + 0.4 * gb_pred
                age_costs.append(cost)
            
            fig = px.line(
                x=list(age_range), 
                y=age_costs,
                title=f"Insurance Cost vs Age (Your Profile)",
                labels={'x': 'Age', 'y': 'Annual Cost ($)'},
                line_shape='linear'
            )
            fig.add_vline(x=age, line_dash="dash", line_color="red", 
                          annotation_text=f"Your Age ({age})")
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tab2:
            # BMI impact analysis
            bmi_range = np.arange(18, 45, 1)
            bmi_costs = []
            
            for test_bmi in bmi_range:
                test_input = np.array([[age, test_bmi, children, sex_encoded, smoker_encoded, region_encoded]])
                test_input_scaled = scaler.transform(test_input)
                if best_model != 'Ensemble':
                    cost = model_info['models'][best_model].predict(test_input_scaled)[0]
                else:
                    lr_pred = model_info['models']['Linear Regression'].predict(test_input_scaled)[0]
                    rf_pred = model_info['models']['Random Forest'].predict(test_input_scaled)[0]
                    gb_pred = model_info['models']['Gradient Boosting'].predict(test_input_scaled)[0]
                    cost = 0.3 * lr_pred + 0.3 * rf_pred + 0.4 * gb_pred
                bmi_costs.append(cost)
            
            fig = px.line(
                x=list(bmi_range), 
                y=bmi_costs,
                title="Insurance Cost vs BMI (Your Profile)",
                labels={'x': 'BMI', 'y': 'Annual Cost ($)'}
            )
            fig.add_vline(x=bmi, line_dash="dash", line_color="red", 
                          annotation_text=f"Your BMI ({bmi:.1f})")
            fig.add_vline(x=25, line_dash="dot", line_color="orange", 
                          annotation_text="Overweight (25)")
            fig.add_vline(x=30, line_dash="dot", line_color="red", 
                          annotation_text="Obese (30)")
            fig.update_traces(line=dict(width=3))
            st.plotly_chart(fig, use_container_width=True)
        
        with analysis_tab3:
            # Smoking impact
            smoking_options = data['smoker'].unique()
            smoking_costs = []
            smoking_labels = []
            
            for smoke_status in smoking_options:
                smoke_encoded = label_encoders['smoker'].transform([smoke_status])[0]
                test_input = np.array([[age, bmi, children, sex_encoded, smoke_encoded, region_encoded]])
                test_input_scaled = scaler.transform(test_input)
                if best_model != 'Ensemble':
                    cost = model_info['models'][best_model].predict(test_input_scaled)[0]
                else:
                    lr_pred = model_info['models']['Linear Regression'].predict(test_input_scaled)[0]
                    rf_pred = model_info['models']['Random Forest'].predict(test_input_scaled)[0]
                    gb_pred = model_info['models']['Gradient Boosting'].predict(test_input_scaled)[0]
                    cost = 0.3 * lr_pred + 0.3 * rf_pred + 0.4 * gb_pred
                smoking_costs.append(cost)
                smoking_labels.append(smoke_status.title())
            
            fig = px.bar(
                x=smoking_labels, 
                y=smoking_costs,
                title="Cost Impact of Smoking Status",
                labels={'x': 'Smoking Status', 'y': 'Annual Cost ($)'},
                color=smoking_costs,
                color_continuous_scale='Reds',
                text=[f"${cost:,.0f}" for cost in smoking_costs]
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            if len(smoking_costs) == 2:
                diff = max(smoking_costs) - min(smoking_costs)
                pct_increase = (diff / min(smoking_costs)) * 100
                st.info(f"💡 **Smoking Impact:** Increases cost by ${diff:,.0f} ({pct_increase:.0f}% increase)")
        
        with analysis_tab4:
            # Regional comparison
            regions = data['region'].unique()
            region_costs = []
            
            for test_region in regions:
                region_enc = label_encoders['region'].transform([test_region])[0]
                test_input = np.array([[age, bmi, children, sex_encoded, smoker_encoded, region_enc]])
                test_input_scaled = scaler.transform(test_input)
                if best_model != 'Ensemble':
                    cost = model_info['models'][best_model].predict(test_input_scaled)[0]
                else:
                    lr_pred = model_info['models']['Linear Regression'].predict(test_input_scaled)[0]
                    rf_pred = model_info['models']['Random Forest'].predict(test_input_scaled)[0]
                    gb_pred = model_info['models']['Gradient Boosting'].predict(test_input_scaled)[0]
                    cost = 0.3 * lr_pred + 0.3 * rf_pred + 0.4 * gb_pred
                region_costs.append(cost)
            
            fig = px.bar(
                x=regions, 
                y=region_costs,
                title="Insurance Costs by Region (Your Profile)",
                labels={'x': 'Region', 'y': 'Annual Cost ($)'},
                color=region_costs,
                color_continuous_scale='Blues',
                text=[f"${cost:,.0f}" for cost in region_costs]
            )
            fig.update_traces(textposition='outside')
            
            # Highlight user's region
            user_region_idx = list(regions).index(region)
            fig.add_annotation(
                x=user_region_idx, 
                y=region_costs[user_region_idx] + 500,
                text="Your Region", 
                showarrow=True, 
                arrowhead=2,
                arrowcolor="red",
                font=dict(color="red", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors and recommendations
        st.markdown('<h2 class="sub-header">🎯 Risk Factors & Recommendations</h2>', unsafe_allow_html=True)
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.markdown("### 🔴 High-Cost Risk Factors")
            high_risks = []
            
            # Analyze user's risk factors
            if smoker.lower() in ['yes', 'true', '1']:
                high_risks.append("🚫 **Smoking**: Major cost driver")
            if bmi > 30:
                high_risks.append("⚖️ **Obesity**: BMI > 30 increases medical risks")
            if age > 50:
                high_risks.append("📈 **Age**: Healthcare costs typically increase with age")
            
            if high_risks:
                for risk in high_risks:
                    st.markdown(f"• {risk}")
            else:
                st.success("✅ No major high-cost risk factors detected!")
        
        with risk_col2:
            st.markdown("### 🟢 Positive Health Factors")
            positive_factors = []
            
            if smoker.lower() in ['no', 'false', '0']:
                positive_factors.append("✅ **Non-smoker**: Significantly reduces costs")
            if 18.5 <= bmi <= 25:
                positive_factors.append("✅ **Healthy BMI**: Optimal weight range")
            if age < 35:
                positive_factors.append("✅ **Young Age**: Generally lower healthcare needs")
            
            for factor in positive_factors:
                st.markdown(f"• {factor}")
            
            if not positive_factors:
                st.info("💡 Consider lifestyle changes to reduce insurance costs")
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check your input values and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
    <h3>📋 Important Disclaimers</h3>
    <ul>
    <li><strong>Educational Purpose:</strong> This prediction is for educational and informational purposes only</li>
    <li><strong>Not Official Quote:</strong> For actual insurance quotes, please consult licensed insurance providers</li>
    <li><strong>Model Limitations:</strong> Predictions are based on historical data and may not reflect current market conditions</li>
    <li><strong>Individual Variation:</strong> Actual costs may vary based on specific policy terms, medical history, and other factors</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if model_info:
        best_r2 = model_info['results'][model_info['best_model']]['r2']
        best_rmse = model_info['results'][model_info['best_model']]['rmse']
        
        st.markdown(f"""
        **🔬 Technical Details:**
        - **Best Model:** {model_info['best_model']} (R² = {best_r2:.3f})
        - **Dataset Size:** {len(data):,} insurance records
        - **Model Accuracy:** {best_r2*100:.1f}% of cost variation explained
        - **Prediction Error (RMSE):** ${best_rmse:,.0f}
        - **Features Used:** Age, BMI, Children, Gender, Smoking Status, Region
        
        Made with ❤️ using Streamlit, scikit-learn, and Plotly | Data Source: Medical Insurance Dataset
        """)

if __name__ == "__main__":
    main()
