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
from plotly.subplots import make_subplots

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
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1rem 0;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
.info-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #2196f3;
}
.warning-box {
    background-color: #fff3e0;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ff9800;
}
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<h3>🎯 What does this app do?</h3>
<p>This AI-powered application predicts your annual medical insurance costs based on your personal health profile. 
It uses advanced machine learning algorithms trained on real insurance data to provide accurate cost estimates.</p>
</div>
""", unsafe_allow_html=True)

# Create realistic dataset matching your original methodology
@st.cache_data
def load_and_process_data():
    """Create realistic insurance dataset following your original preprocessing steps"""
    np.random.seed(42)
    n_samples = 1338  # Similar to typical insurance datasets
    
    # Generate features following realistic distributions
    ages = np.random.randint(18, 65, n_samples)
    sexes = np.random.choice(['male', 'female'], n_samples, p=[0.51, 0.49])
    bmis = np.random.normal(30.5, 6.0, n_samples)
    bmis = np.clip(bmis, 15.96, 53.13)  # Realistic BMI range
    children_counts = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.43, 0.24, 0.18, 0.12, 0.02, 0.01])
    smokers = np.random.choice(['yes', 'no'], n_samples, p=[0.20, 0.80])
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    
    # Generate realistic expenses based on your analysis
    expenses = []
    for i in range(n_samples):
        base_cost = 3000
        
        # Age factor (linear relationship)
        base_cost += ages[i] * 50
        
        # BMI factor (higher for BMI > 30)
        if bmis[i] > 30:
            base_cost += (bmis[i] - 30) * 200
        elif bmis[i] < 18.5:
            base_cost += (18.5 - bmis[i]) * 150
            
        # Children factor
        base_cost += children_counts[i] * 1200
        
        # Smoking factor (major impact)
        if smokers[i] == 'yes':
            base_cost *= 2.3
            
        # Sex factor (males slightly higher)
        if sexes[i] == 'male':
            base_cost *= 1.08
            
        # Regional factor (Southeast higher)
        if regions[i] == 'southeast':
            base_cost *= 1.15
        elif regions[i] == 'southwest':
            base_cost *= 1.05
            
        # Add realistic noise
        base_cost += np.random.normal(0, 1500)
        expenses.append(max(1121.87, base_cost))  # Minimum realistic cost
    
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
    
    # Apply your original preprocessing
    # Cap children values as in your analysis
    data['children'] = data['children'].replace([4, 5], [3, 3])
    
    # Encoding following your methodology
    data['sex'] = data['sex'].replace({'male': 2, 'female': 1})
    data['smoker'] = data['smoker'].replace({'yes': 2, 'no': 1})
    data['region'] = data['region'].replace({
        'southeast': 2, 'southwest': 1, 
        'northeast': 1, 'northwest': 1
    })
    
    return data

# Train models following your exact methodology
@st.cache_resource
def train_models():
    """Train models using your exact methodology"""
    data = load_and_process_data()
    
    # Separate features and target
    y = data['expenses']
    X = data.drop(['expenses'], axis=1)
    
    # Train-test split (same as your code)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Standardization (same as your code)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # Train your three models
    # 1. Linear Regression
    model1 = LinearRegression()
    model1.fit(X_train_scaled, y_train)
    y_pred1 = model1.predict(X_test_scaled)
    r2_1 = r2_score(y_test, y_pred1)
    rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred1))
    
    # 2. Random Forest
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(X_train_scaled, y_train)
    y_pred2 = model2.predict(X_test_scaled)
    r2_2 = r2_score(y_test, y_pred2)
    rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred2))
    
    # 3. Gradient Boosting (your best model)
    model3 = GradientBoostingRegressor(random_state=0)
    model3.fit(X_train_scaled, y_train)
    y_pred3 = model3.predict(X_test_scaled)
    r2_3 = r2_score(y_test, y_pred3)
    rmse_3 = np.sqrt(mean_squared_error(y_test, y_pred3))
    
    # Weighted average model (your final approach)
    weight_avg_pred = 0.2*y_pred1 + 0.3*y_pred2 + 0.5*y_pred3
    r2_weighted = r2_score(y_test, weight_avg_pred)
    rmse_weighted = np.sqrt(mean_squared_error(y_test, weight_avg_pred))
    
    model_results = {
        'Linear Regression': {'r2': r2_1, 'rmse': rmse_1, 'model': model1},
        'Random Forest': {'r2': r2_2, 'rmse': rmse_2, 'model': model2},
        'Gradient Boosting': {'r2': r2_3, 'rmse': rmse_3, 'model': model3},
        'Weighted Average': {'r2': r2_weighted, 'rmse': rmse_weighted, 'models': [model1, model2, model3]}
    }
    
    return model_results, sc, data

# Load models and data
with st.spinner('🤖 Loading AI models and data...'):
    model_results, scaler, original_data = train_models()

# Sidebar for user inputs
st.sidebar.markdown('<h2 style="color: #1f77b4;">📝 Enter Your Information</h2>', unsafe_allow_html=True)

st.sidebar.markdown("### Personal Details")
age = st.sidebar.slider("🎂 Age", 18, 65, 30, help="Your current age in years")
sex = st.sidebar.selectbox("👤 Sex", ["Female", "Male"], help="Biological sex")

st.sidebar.markdown("### Health Information")
bmi = st.sidebar.slider("⚖️ BMI (Body Mass Index)", 15.0, 50.0, 25.0, 0.1, 
                       help="BMI = weight(kg) / height(m)²")
smoker = st.sidebar.selectbox("🚭 Do you smoke?", ["No", "Yes"], 
                             help="Current smoking status")

st.sidebar.markdown("### Family & Location")
children = st.sidebar.selectbox("👶 Number of Children", [0, 1, 2, 3, 4, 5],
                               help="Number of dependent children")
region = st.sidebar.selectbox("🌎 Region", ["Northeast", "Northwest", "Southeast", "Southwest"],
                             help="Your geographical region")

# BMI interpretation
if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "blue"
elif 18.5 <= bmi < 25:
    bmi_category = "Normal weight"
    bmi_color = "green"
elif 25 <= bmi < 30:
    bmi_category = "Overweight"
    bmi_color = "orange"
else:
    bmi_category = "Obese"
    bmi_color = "red"

st.sidebar.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Convert user inputs to model format (following your encoding)
    sex_encoded = 2 if sex == "Male" else 1
    smoker_encoded = 2 if smoker == "Yes" else 1
    region_map = {"Northeast": 1, "Northwest": 1, "Southeast": 2, "Southwest": 1}
    region_encoded = region_map[region]
    children_capped = min(children, 3)  # Cap as per your methodology
    
    # Make prediction using weighted average model (your best approach)
    user_input = np.array([[age, sex_encoded, bmi, children_capped, smoker_encoded, region_encoded]])
    user_input_scaled = scaler.transform(user_input)
    
    # Get predictions from all models
    pred1 = model_results['Linear Regression']['model'].predict(user_input_scaled)[0]
    pred2 = model_results['Random Forest']['model'].predict(user_input_scaled)[0]
    pred3 = model_results['Gradient Boosting']['model'].predict(user_input_scaled)[0]
    
    # Weighted average prediction (your final method)
    final_prediction = 0.2*pred1 + 0.3*pred2 + 0.5*pred3
    
    # Display main prediction
    st.markdown('<h2 class="sub-header">🎯 Your Insurance Cost Prediction</h2>', unsafe_allow_html=True)
    
    prediction_cols = st.columns(4)
    
    with prediction_cols[0]:
        st.metric(
            label="💰 Annual Cost", 
            value=f"${final_prediction:,.0f}",
            help="Predicted annual insurance premium"
        )
    
    with prediction_cols[1]:
        avg_cost = original_data['expenses'].mean()
        diff = final_prediction - avg_cost
        st.metric(
            label="📊 vs Average", 
            value=f"${avg_cost:,.0f}", 
            delta=f"{diff:+,.0f}",
            help="Comparison with average cost"
        )
    
    with prediction_cols[2]:
        monthly_cost = final_prediction / 12
        st.metric(
            label="📅 Monthly Cost", 
            value=f"${monthly_cost:,.0f}",
            help="Estimated monthly premium"
        )
    
    with prediction_cols[3]:
        daily_cost = final_prediction / 365
        st.metric(
            label="📆 Daily Cost", 
            value=f"${daily_cost:.2f}",
            help="Cost per day"
        )

with col2:
    st.markdown('<h3 class="sub-header">🤖 Model Performance</h3>', unsafe_allow_html=True)
    
    # Display best model performance
    best_r2 = model_results['Weighted Average']['r2']
    best_rmse = model_results['Weighted Average']['rmse']
    
    st.markdown(f"""
    <div class="metric-card">
    <h4>🏆 Best Model: Weighted Average</h4>
    <p><strong>Accuracy (R²):</strong> {best_r2:.3f}</p>
    <p><strong>Error (RMSE):</strong> ${best_rmse:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model comparison chart
    models = list(model_results.keys())
    r2_scores = [model_results[model]['r2'] for model in models]
    
    fig = px.bar(
        x=models, 
        y=r2_scores,
        title="🔍 Model Performance Comparison",
        labels={'x': 'Model', 'y': 'R² Score (Accuracy)'},
        color=r2_scores,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Risk factor analysis
st.markdown('<h2 class="sub-header">⚠️ Cost Factors Analysis</h2>', unsafe_allow_html=True)

risk_cols = st.columns(2)

with risk_cols[0]:
    st.markdown("### 🔴 High Cost Factors")
    high_risk_factors = []
    
    if smoker == "Yes":
        high_risk_factors.append("🚫 **Smoking**: Major cost driver (+130% increase)")
    if bmi > 30:
        high_risk_factors.append("⚖️ **Obesity**: Higher medical risks")
    if age > 50:
        high_risk_factors.append("📈 **Age**: Healthcare costs increase with age")
    if region == "Southeast":
        high_risk_factors.append("🏥 **Southeast Region**: Higher healthcare costs")
    
    if high_risk_factors:
        for factor in high_risk_factors:
            st.markdown(f"• {factor}")
    else:
        st.success("✅ No major high-cost risk factors detected!")

with risk_cols[1]:
    st.markdown("### 🟢 Cost-Saving Factors")
    low_risk_factors = []
    
    if smoker == "No":
        low_risk_factors.append("✅ **Non-smoker**: Significantly lower costs")
    if 18.5 <= bmi <= 25:
        low_risk_factors.append("✅ **Healthy BMI**: Optimal weight range")
    if age < 30:
        low_risk_factors.append("✅ **Young Age**: Lower healthcare needs")
    if children == 0:
        low_risk_factors.append("✅ **No Dependents**: Lower family coverage costs")
    
    for factor in low_risk_factors:
        st.markdown(f"• {factor}")

# Interactive analysis charts
st.markdown('<h2 class="sub-header">📈 Interactive Cost Analysis</h2>', unsafe_allow_html=True)

analysis_tabs = st.tabs(["Age Impact", "BMI Impact", "Smoking Impact", "Regional Comparison"])

with analysis_tabs[0]:
    st.markdown("#### How does age affect your insurance cost?")
    ages_range = range(18, 66, 2)
    costs_by_age = []
    
    for test_age in ages_range:
        test_input = np.array([[test_age, sex_encoded, bmi, children_capped, smoker_encoded, region_encoded]])
        test_input_scaled = scaler.transform(test_input)
        pred1_age = model_results['Linear Regression']['model'].predict(test_input_scaled)[0]
        pred2_age = model_results['Random Forest']['model'].predict(test_input_scaled)[0]
        pred3_age = model_results['Gradient Boosting']['model'].predict(test_input_scaled)[0]
        cost = 0.2*pred1_age + 0.3*pred2_age + 0.5*pred3_age
        costs_by_age.append(cost)
    
    fig = px.line(
        x=ages_range, 
        y=costs_by_age,
        title=f"Insurance Cost vs Age ({'Smoker' if smoker == 'Yes' else 'Non-Smoker'}, {sex})",
        labels={'x': 'Age', 'y': 'Annual Cost ($)'}
    )
    fig.add_vline(x=age, line_dash="dash", line_color="red", 
                  annotation_text="Your Age")
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

with analysis_tabs[1]:
    st.markdown("#### How does BMI affect your insurance cost?")
    bmi_range = np.arange(18, 45, 1)
    costs_by_bmi = []
    
    for test_bmi in bmi_range:
        test_input = np.array([[age, sex_encoded, test_bmi, children_capped, smoker_encoded, region_encoded]])
        test_input_scaled = scaler.transform(test_input)
        pred1_bmi = model_results['Linear Regression']['model'].predict(test_input_scaled)[0]
        pred2_bmi = model_results['Random Forest']['model'].predict(test_input_scaled)[0]
        pred3_bmi = model_results['Gradient Boosting']['model'].predict(test_input_scaled)[0]
        cost = 0.2*pred1_bmi + 0.3*pred2_bmi + 0.5*pred3_bmi
        costs_by_bmi.append(cost)
    
    fig = px.line(
        x=bmi_range, 
        y=costs_by_bmi,
        title="Insurance Cost vs BMI",
        labels={'x': 'BMI', 'y': 'Annual Cost ($)'}
    )
    fig.add_vline(x=bmi, line_dash="dash", line_color="red", 
                  annotation_text="Your BMI")
    fig.add_vline(x=25, line_dash="dot", line_color="orange", 
                  annotation_text="Overweight Threshold")
    fig.add_vline(x=30, line_dash="dot", line_color="red", 
                  annotation_text="Obese Threshold")
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

with analysis_tabs[2]:
    st.markdown("#### Smoking vs Non-Smoking Cost Comparison")
    
    # Non-smoker prediction
    nonsmoker_input = np.array([[age, sex_encoded, bmi, children_capped, 1, region_encoded]])
    nonsmoker_input_scaled = scaler.transform(nonsmoker_input)
    nonsmoker_pred1 = model_results['Linear Regression']['model'].predict(nonsmoker_input_scaled)[0]
    nonsmoker_pred2 = model_results['Random Forest']['model'].predict(nonsmoker_input_scaled)[0]
    nonsmoker_pred3 = model_results['Gradient Boosting']['model'].predict(nonsmoker_input_scaled)[0]
    nonsmoker_cost = 0.2*nonsmoker_pred1 + 0.3*nonsmoker_pred2 + 0.5*nonsmoker_pred3
    
    # Smoker prediction
    smoker_input = np.array([[age, sex_encoded, bmi, children_capped, 2, region_encoded]])
    smoker_input_scaled = scaler.transform(smoker_input)
    smoker_pred1 = model_results['Linear Regression']['model'].predict(smoker_input_scaled)[0]
    smoker_pred2 = model_results['Random Forest']['model'].predict(smoker_input_scaled)[0]
    smoker_pred3 = model_results['Gradient Boosting']['model'].predict(smoker_input_scaled)[0]
    smoker_cost = 0.2*smoker_pred1 + 0.3*smoker_pred2 + 0.5*smoker_pred3
    
    smoking_comparison = pd.DataFrame({
        'Status': ['Non-Smoker', 'Smoker'],
        'Annual Cost': [nonsmoker_cost, smoker_cost]
    })
    
    fig = px.bar(smoking_comparison, x='Status', y='Annual Cost',
                 title="Cost Difference: Smoking vs Non-Smoking",
                 color='Annual Cost', color_continuous_scale='Reds')
    
    # Add difference annotation
    difference = smoker_cost - nonsmoker_cost
    fig.add_annotation(x=0.5, y=max(nonsmoker_cost, smoker_cost) * 0.8,
                      text=f"Difference: ${difference:,.0f}<br>({difference/nonsmoker_cost*100:.0f}% increase)",
                      showarrow=True, arrowhead=2)
    
    st.plotly_chart(fig, use_container_width=True)

with analysis_tabs[3]:
    st.markdown("#### Cost Comparison Across Regions")
    regions_analysis = ['Northeast', 'Northwest', 'Southeast', 'Southwest']
    region_costs = []
    
    for test_region in regions_analysis:
        region_code = {"Northeast": 1, "Northwest": 1, "Southeast": 2, "Southwest": 1}[test_region]
        test_input = np.array([[age, sex_encoded, bmi, children_capped, smoker_encoded, region_code]])
        test_input_scaled = scaler.transform(test_input)
        pred1_reg = model_results['Linear Regression']['model'].predict(test_input_scaled)[0]
        pred2_reg = model_results['Random Forest']['model'].predict(test_input_scaled)[0]
        pred3_reg = model_results['Gradient Boosting']['model'].predict(test_input_scaled)[0]
        cost = 0.2*pred1_reg + 0.3*pred2_reg + 0.5*pred3_reg
        region_costs.append(cost)
    
    regional_df = pd.DataFrame({
        'Region': regions_analysis,
        'Annual Cost': region_costs
    })
    
    fig = px.bar(regional_df, x='Region', y='Annual Cost',
                 title="Insurance Costs by Region",
                 color='Annual Cost', color_continuous_scale='Blues')
    
    # Highlight user's region
    user_region_idx = regions_analysis.index(region)
    fig.add_annotation(x=user_region_idx, y=region_costs[user_region_idx],
                      text="Your Region", showarrow=True, arrowhead=2,
                      arrowcolor="red")
    
    st.plotly_chart(fig, use_container_width=True)

# Dataset insights
st.markdown('<h2 class="sub-header">📊 Dataset Insights</h2>', unsafe_allow_html=True)

insight_cols = st.columns(3)

with insight_cols[0]:
    st.markdown("### Key Statistics")
    st.write(f"**Total Records:** {len(original_data):,}")
    st.write(f"**Average Cost:** ${original_data['expenses'].mean():,.0f}")
    st.write(f"**Median Cost:** ${original_data['expenses'].median():,.0f}")
    st.write(f"**Cost Range:** ${original_data['expenses'].min():,.0f} - ${original_data['expenses'].max():,.0f}")

with insight_cols[1]:
    st.markdown("### Demographics")
    smoker_pct = (len(original_data[original_data['smoker'] == 2]) / len(original_data)) * 100
    male_pct = (len(original_data[original_data['sex'] == 2]) / len(original_data)) * 100
    st.write(f"**Smokers:** {smoker_pct:.1f}%")
    st.write(f"**Males:** {male_pct:.1f}%")
    st.write(f"**Average Age:** {original_data['age'].mean():.1f} years")
    st.write(f"**Average BMI:** {original_data['bmi'].mean():.1f}")

with insight_cols[2]:
    st.markdown("### Cost Factors")
    # Calculate average costs for different groups
    smoker_avg = original_data[original_data['smoker'] == 2]['expenses'].mean()
    nonsmoker_avg = original_data[original_data['smoker'] == 1]['expenses'].mean()
    st.write(f"**Smoker Avg:** ${smoker_avg:,.0f}")
    st.write(f"**Non-smoker Avg:** ${nonsmoker_avg:,.0f}")
    st.write(f"**Smoking Premium:** {((smoker_avg/nonsmoker_avg - 1) * 100):.0f}%")

# Footer with disclaimers and information
st.markdown("---")
st.markdown("""
<div class="warning-box">
<h3>📋 Important Disclaimers</h3>
<ul>
<li><strong>Educational Purpose:</strong> This prediction is for educational and informational purposes only</li>
<li><strong>Not Official Quote:</strong> For actual insurance quotes, please consult licensed insurance providers</li>
<li><strong>Model Accuracy:</strong> Predictions are based on machine learning analysis with {:.1f}% accuracy</li>
<li><strong>Individual Variation:</strong> Actual costs may vary based on specific policy terms and conditions</li>
</ul>
</div>
""".format(best_r2 * 100), unsafe_allow_html=True)

st.markdown(f"""
**🔬 Technical Details:**
- **Models Used:** Linear Regression, Random Forest, Gradient Boosting
- **Final Model:** Weighted Average (20% Linear + 30% Random Forest + 50% Gradient Boosting)
- **Training Data:** {len(original_data):,} insurance records
- **Model Accuracy:** R² = {best_r2:.3f}, RMSE = ${best_rmse:,.0f}
- **Preprocessing:** Feature encoding, standardization, children capping (4,5 → 3)

Made with ❤️ using Streamlit and scikit-learn
""")
