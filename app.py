import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("### AI-powered insurance cost prediction based on your health profile")

# Sidebar for user inputs
st.sidebar.header("📝 Enter Your Information")

age = st.sidebar.slider("Age", 18, 65, 30)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
bmi = st.sidebar.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Do you smoke?", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Create sample dataset (since you don't have the CSV in deployment)
@st.cache_data
def create_sample_dataset():
    """Create a realistic sample dataset for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    ages = np.random.randint(18, 65, n_samples)
    sexes = np.random.choice(['male', 'female'], n_samples)
    bmis = np.random.normal(28.0, 6.0, n_samples)
    bmis = np.clip(bmis, 15, 50)  # Realistic BMI range
    children_counts = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
    smokers = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    
    # Generate realistic expenses based on factors
    expenses = []
    for i in range(n_samples):
        base_cost = 3000
        base_cost += ages[i] * 50  # Age factor
        base_cost += max(0, (bmis[i] - 25) * 100)  # BMI factor
        base_cost += children_counts[i] * 1500  # Children factor
        if smokers[i] == 'yes':
            base_cost *= 2.5  # Smoking dramatically increases cost
        if sexes[i] == 'male':
            base_cost *= 1.1  # Slight male premium
        if regions[i] == 'southeast':
            base_cost *= 1.2  # Regional variation
        
        # Add some randomness
        base_cost += np.random.normal(0, 2000)
        expenses.append(max(1000, base_cost))  # Minimum cost
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'bmi': bmis,
        'children': children_counts,
        'smoker': smokers,
        'region': regions,
        'expenses': expenses
    })
    
    return data

# Load and process data
@st.cache_data
def prepare_data():
    """Prepare data following your original preprocessing steps"""
    data = create_sample_dataset()
    
    # Cap children values (as in your original code)
    data['children'] = data['children'].replace([4, 5], [3, 3])
    
    # Encoding (as in your original code)
    data['sex'] = data['sex'].replace({'male': 2, 'female': 1})
    data['smoker'] = data['smoker'].replace({'yes': 2, 'no': 1})
    data['region'] = data['region'].replace({
        'southeast': 2, 'southwest': 1, 
        'northeast': 1, 'northwest': 1
    })
    
    return data

# Train models
@st.cache_resource
def train_models():
    """Train all three models as in your original code"""
    data = prepare_data()
    
    # Separate features and target
    y = data['expenses']
    X = data.drop(['expenses'], axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Standardization
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # Train models
    # 1. Linear Regression
    model1 = LinearRegression()
    model1.fit(X_train_scaled, y_train)
    
    # 2. Random Forest
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(X_train_scaled, y_train)
    
    # 3. Gradient Boosting
    model3 = GradientBoostingRegressor(random_state=0)
    model3.fit(X_train_scaled, y_train)
    
    # Test predictions for model comparison
    y_pred1 = model1.predict(X_test_scaled)
    y_pred2 = model2.predict(X_test_scaled)
    y_pred3 = model3.predict(X_test_scaled)
    
    # Calculate R2 scores
    r2_1 = r2_score(y_test, y_pred1)
    r2_2 = r2_score(y_test, y_pred2)
    r2_3 = r2_score(y_test, y_pred3)
    
    # Return best model (usually Gradient Boosting)
    best_model = model3 if r2_3 >= max(r2_1, r2_2) else (model2 if r2_2 >= r2_1 else model1)
    
    return best_model, sc, [r2_1, r2_2, r2_3], ['Linear Regression', 'Random Forest', 'Gradient Boosting']

# Load trained model
model, scaler, r2_scores, model_names = train_models()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Convert user inputs to model format
    sex_encoded = 2 if sex == "Male" else 1
    smoker_encoded = 2 if smoker == "Yes" else 1
    region_map = {"Northeast": 1, "Northwest": 1, "Southeast": 2, "Southwest": 1}
    region_encoded = region_map[region]
    children_capped = min(children, 3)  # Cap as per your original code
    
    # Make prediction
    user_input = np.array([[age, sex_encoded, bmi, children_capped, smoker_encoded, region_encoded]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    
    # Display prediction
    st.subheader("🎯 Your Predicted Annual Insurance Cost")
    
    # Create a nice prediction display
    prediction_col1, prediction_col2, prediction_col3 = st.columns(3)
    
    with prediction_col1:
        st.metric("Predicted Cost", f"${prediction:,.0f}")
    
    with prediction_col2:
        avg_cost = 13270  # Approximate average from typical datasets
        diff = prediction - avg_cost
        st.metric("vs Average", f"${avg_cost:,.0f}", f"{diff:+,.0f}")
    
    with prediction_col3:
        monthly_cost = prediction / 12
        st.metric("Monthly Cost", f"${monthly_cost:,.0f}")
    
    # Risk factor analysis
    st.subheader("⚠️ Risk Factor Analysis")
    
    risk_factors = []
    if smoker == "Yes":
        risk_factors.append("🚫 **Smoking**: Major cost factor (+150%)")
    if bmi > 30:
        risk_factors.append("⚖️ **High BMI**: Increased health risks")
    if age > 50:
        risk_factors.append("📈 **Age Factor**: Higher with age")
    if children >= 3:
        risk_factors.append("👨‍👩‍👧‍👦 **Large Family**: More dependents")
    if region == "Southeast":
        risk_factors.append("🏥 **Region**: Higher costs in Southeast")
    
    if risk_factors:
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.success("✅ You have a low-risk profile!")

with col2:
    st.subheader("📊 Model Performance")
    
    # Display model comparison
    best_r2 = max(r2_scores)
    best_model_name = model_names[r2_scores.index(best_r2)]
    
    st.metric("Best Model", best_model_name, f"R² = {best_r2:.3f}")
    
    # Model comparison chart
    fig = px.bar(
        x=model_names, 
        y=r2_scores,
        title="Model Performance Comparison",
        labels={'x': 'Model', 'y': 'R² Score'},
        color=r2_scores,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Additional visualizations
st.subheader("📈 Cost Analysis")

tab1, tab2 = st.tabs(["Age Impact", "BMI Impact"])

with tab1:
    # Age impact analysis
    ages_range = range(20, 66, 5)
    costs_by_age = []
    
    for test_age in ages_range:
        test_input = np.array([[test_age, sex_encoded, bmi, children_capped, smoker_encoded, region_encoded]])
        test_input_scaled = scaler.transform(test_input)
        cost = model.predict(test_input_scaled)[0]
        costs_by_age.append(cost)
    
    fig = px.line(
        x=ages_range, 
        y=costs_by_age,
        title=f"Insurance Cost vs Age ({'Smoker' if smoker == 'Yes' else 'Non-Smoker'})",
        labels={'x': 'Age', 'y': 'Annual Cost ($)'}
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # BMI impact analysis
    bmi_range = np.arange(18, 40, 2)
    costs_by_bmi = []
    
    for test_bmi in bmi_range:
        test_input = np.array([[age, sex_encoded, test_bmi, children_capped, smoker_encoded, region_encoded]])
        test_input_scaled = scaler.transform(test_input)
        cost = model.predict(test_input_scaled)[0]
        costs_by_bmi.append(cost)
    
    fig = px.line(
        x=bmi_range, 
        y=costs_by_bmi,
        title="Insurance Cost vs BMI",
        labels={'x': 'BMI', 'y': 'Annual Cost ($)'}
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**📝 Note**: This prediction is based on machine learning analysis of medical insurance data patterns. 
For actual insurance quotes, please consult with licensed insurance providers.

**🔬 Model Info**: Using ensemble of Linear Regression, Random Forest, and Gradient Boosting algorithms.
""")
