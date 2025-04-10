import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings
import joblib
import numpy as np
from supabase import create_client, Client

# Ignore warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Attrition Dashboard & Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# --- Supabase Connection ---


@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError:
        st.error(
            "Supabase credentials not found in Streamlit secrets. Please add SUPABASE_URL and SUPABASE_KEY.")
        return None
    except Exception as e:
        st.error(f"Error initializing Supabase connection: {e}")
        return None


supabase = init_connection()

# --- Model Loading ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    SCRIPT_DIR, 'model', 'logistic_regression_pipeline.joblib')


@st.cache_resource
def load_model_pipeline(path):
    try:
        model_pipeline = joblib.load(path)
        # Extract feature names from the preprocessor if possible
        try:
            preprocessor = model_pipeline.named_steps['preprocessor']
            num_features = [
                t[2][i] for t in preprocessor.transformers_ if t[0] == 'num' for i in range(len(t[2]))]
            cat_features = [
                t[2][i] for t in preprocessor.transformers_ if t[0] == 'cat' for i in range(len(t[2]))]
            feature_names = num_features + cat_features
        except Exception as e:
            st.warning(
                f"Could not automatically determine feature names from pipeline: {e}. Manual list will be used.")
            # Define features manually if extraction fails
            feature_names = [
                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                'MaritalStatus', 'OverTime'
            ]  # Example
        return model_pipeline, feature_names
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}")
        st.error(
            "Please ensure 'logistic_regression_pipeline.joblib' is in the 'model' subdirectory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading the model pipeline: {e}")
        return None, None


model_pipeline, feature_names = load_model_pipeline(MODEL_PATH)


# --- Data Loading and Cleaning Function (from Supabase) ---
@st.cache_data(ttl=600)
def load_data():
    if supabase is None:
        st.error("Supabase connection is not available. Cannot load data.")
        return None
    try:
        # Query from Supabase table
        response = supabase.table(
            'cleaned_employee_data').select("*").execute()

        if not response.data:
            st.error(
                "Failed to fetch data from Supabase or table 'cleaned_employee_data' is empty.")
            st.info(
                "Please ensure the table exists, has data, and Row Level Security (RLS) allows reads for the 'anon' role.")
            return None

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(response.data)
        # Debug confirmation
        st.success(
            f"Successfully loaded {len(df)} rows from Supabase table 'cleaned_employee_data'.")

        # --- Data Cleaning/Transformation ---
        numeric_cols = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Attrition'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(
                    f"Expected numeric column '{col}' not found in Supabase data.")

        if 'Attrition' not in df.columns:
            st.error(
                "Error: 'Attrition' column not found in the data from Supabase.")
            return None

        df_cleaned = df.dropna(subset=['Attrition']).copy()

        if df_cleaned.empty:
            st.error(
                "Dataframe is empty after dropping rows with missing Attrition.")
            return None

        # Ensure Attrition is integer after potential float conversion
        df_cleaned['Attrition'] = df_cleaned['Attrition'].astype(int)

        # --- Feature Engineering ---
        df_cleaned['Attrition_Status'] = df_cleaned['Attrition'].map(
            {0: 'No', 1: 'Yes'})

        satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        for col in ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']:
            if col in df_cleaned.columns:
                # Check if column is numeric before mapping
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[f'{col}_Label'] = df_cleaned[col].map(
                        satisfaction_map).fillna('Unknown')
                else:
                    st.warning(
                        f"Column '{col}' is not numeric in Supabase data, cannot create label.")
            else:
                st.warning(
                    f"Expected satisfaction column '{col}' not found in Supabase data.")

        worklife_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
        if 'WorkLifeBalance' in df_cleaned.columns:
            if pd.api.types.is_numeric_dtype(df_cleaned['WorkLifeBalance']):
                df_cleaned['WorkLifeBalance_Label'] = df_cleaned['WorkLifeBalance'].map(
                    worklife_map).fillna('Unknown')
            else:
                st.warning(
                    "Column 'WorkLifeBalance' is not numeric in Supabase data, cannot create label.")
        else:
            st.warning(
                "Expected column 'WorkLifeBalance' not found in Supabase data.")

        return df_cleaned

    except Exception as e:
        st.error(
            f"An error occurred connecting to Supabase or processing data: {e}")
        return None


# --- Load Data (uses the Supabase function) ---
df_base = load_data()

# --- Define Single Palette Theme ---
ATTRITION_NO_COLOR = px.colors.sequential.Blues[2]  # Lighter Blue
ATTRITION_YES_COLOR = px.colors.sequential.Blues[5]  # Darker Blue
SEQ_PALETTE = px.colors.sequential.Blues
# Darker blues indicate higher values/rates
SEQ_PALETTE_R = px.colors.sequential.Blues_r

color_map_attrition_single = {
    'No': ATTRITION_NO_COLOR,
    'Yes': ATTRITION_YES_COLOR
}

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:", ["Dashboard Analysis", "Predict Attrition"])

# --- Initialize Session State for Prediction ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None


# ==============================================================================
# --- Helper Functions for Dashboard Analysis ---
# ==============================================================================
def calculate_attrition_rate(df, group_by_col):
    """Calculates attrition rate grouped by a specific column."""
    if group_by_col not in df.columns or 'Attrition' not in df.columns:
        return None
    attrition_data = df.groupby(group_by_col)['Attrition'].agg([
        'mean', 'count']).reset_index()
    attrition_data['Attrition Rate (%)'] = attrition_data['mean'] * 100
    return attrition_data.rename(columns={'mean': 'Attrition Rate (Decimal)', 'count': 'Employee Count'})


def plot_attrition_rate_bar(data, x_col, y_col='Attrition Rate (%)', title="Attrition Rate",
                            color_col=None, color_map=None, color_seq=SEQ_PALETTE_R,
                            sort_order=None, orientation='v', height=400, hover_data=None):
    """Generates a styled bar chart for attrition rates."""
    if data is None or data.empty:
        return None

    rate_col = y_col if orientation == 'v' else x_col

    # --- Data Pre-processing & Sorting ---
    # Drop rows where the rate value is NaN before any processing
    data = data.dropna(subset=[rate_col]).copy()
    if data.empty:
        st.warning(
            f"No valid data found for {title} chart after removing missing rates.")
        return None

    label_col = y_col if orientation == 'h' else x_col

    if sort_order:
        try:
            # Ensure sort_order only contains categories present in the data
            valid_sort_order = [
                cat for cat in sort_order if cat in data[label_col].unique()]
            if not valid_sort_order:
                st.warning(
                    f"Sort order categories not found in data for {title}. Using default.")
                if orientation == 'h':
                    data = data.sort_values(rate_col, ascending=True)
            else:
                data[label_col] = pd.Categorical(
                    data[label_col], categories=valid_sort_order, ordered=True)
                data = data.sort_values(label_col)
        except Exception as e:
            st.warning(
                f"Could not apply sort order to '{label_col}' for {title}. Using default. Error: {e}")
            if orientation == 'h':
                data = data.sort_values(rate_col, ascending=True)
    elif orientation == 'h':
        # Default sort for horizontal bars if no order specified
        data = data.sort_values(rate_col, ascending=True)

    # --- Text Formatting (Handle NaN safely) ---
    text_labels = data[rate_col].apply(
        lambda x: f'{x:.1f}%' if pd.notna(x) else None)

    # --- Plotting ---
    fig = px.bar(data,
                 x=x_col if orientation == 'v' else rate_col,
                 y=y_col if orientation == 'v' else label_col,
                 title=title,
                 color=color_col if color_col else (
                     rate_col if orientation == 'h' else label_col),
                 color_discrete_map=color_map,
                 color_continuous_scale=color_seq if not color_map and orientation == 'h' else None,
                 color_discrete_sequence=color_seq if not color_map and orientation == 'v' else None,
                 text=text_labels,  # Use pre-calculated safe text labels
                 orientation=orientation,
                 height=height,
                 hover_data=hover_data
                 )

    # --- Layout Updates ---
    fig.update_layout(
        yaxis_title=label_col if orientation == 'h' else y_col,
        xaxis_title=rate_col if orientation == 'h' else x_col,
        # Set category order directly here if applicable
        xaxis={'categoryorder': 'array', 'categoryarray': list(
            data[label_col])} if orientation == 'v' and label_col in data else None,
        yaxis={'categoryorder': 'array', 'categoryarray': list(
            data[label_col])} if orientation == 'h' and label_col in data else None,
        coloraxis_showscale=False if orientation == 'h' and not color_map else None,
        title_x=0.5,
        margin=dict(l=150 if orientation == 'h' else 10,
                    r=10, t=50 if title else 30, b=30)
    )
    fig.update_traces(textposition='outside', textfont_size=12)
    return fig


# ==============================================================================
# --- Dashboard Analysis Page ---
# ==============================================================================
if page == "Dashboard Analysis":
    st.session_state.prediction_result = None
    st.session_state.prediction_proba = None

    st.title("📉 Employee Attrition: Key Drivers & Insights")
    st.markdown(
        "Focused analysis for strategic retention efforts at Jaya Jaya Maju.")

    if df_base is None:
        st.error("Data loading failed.")
        st.stop()

# --- Sidebar Filters ---
    st.sidebar.header("Filter Analysis Scope")
    # Allow filtering, but keep it minimal for exec view
    all_departments = sorted(df_base['Department'].unique().tolist())
    selected_departments = st.sidebar.multiselect(
        'Department(s)', all_departments, default=all_departments)

    all_job_levels = sorted(df_base['JobLevel'].unique().tolist())
    selected_job_levels = st.sidebar.multiselect(
        'Job Level(s)', all_job_levels, default=all_job_levels)

    # --- Attribution ---
    st.sidebar.divider()
    st.sidebar.markdown(
        """**Created by: Rasya Radja**
[LinkedIn Profile](https://www.linkedin.com/in/rasyaradja/)"""
    )

    # Apply Filters
    filter_conditions = pd.Series([True] * len(df_base))
    if selected_departments != all_departments:
        filter_conditions &= df_base['Department'].isin(selected_departments)
    if selected_job_levels != all_job_levels:
        filter_conditions &= df_base['JobLevel'].isin(selected_job_levels)

    df_filtered = df_base[filter_conditions].copy()

    if df_filtered.empty:
        st.warning("No data matches filters.")
        st.stop()

    # --- Calculate Key Metrics ---
    total_employees = df_filtered.shape[0]
    attrition_count = int(df_filtered['Attrition'].sum())
    attrition_rate = (attrition_count / total_employees) * \
        100 if total_employees > 0 else 0

    ot_data = calculate_attrition_rate(df_filtered, 'OverTime')
    ot_rate_yes = 0
    ot_rate_no = 0
    if ot_data is not None:
        ot_rate_yes = ot_data.loc[ot_data['OverTime'] == 'Yes',
                                  'Attrition Rate (%)'].iloc[0] if not ot_data[ot_data['OverTime'] == 'Yes'].empty else 0
        ot_rate_no = ot_data.loc[ot_data['OverTime'] == 'No',
                                 'Attrition Rate (%)'].iloc[0] if not ot_data[ot_data['OverTime'] == 'No'].empty else 0
    ot_diff = ot_rate_yes - ot_rate_no

    med_income_stayers = df_filtered[df_filtered['Attrition'] == 0]['MonthlyIncome'].median(
    ) if 'MonthlyIncome' in df_filtered.columns else 0
    med_income_leavers = df_filtered[df_filtered['Attrition'] == 1]['MonthlyIncome'].median(
    ) if 'MonthlyIncome' in df_filtered.columns else 0
    income_gap = med_income_stayers - med_income_leavers

    # --- Header & KPIs ---
    st.markdown(
        f"#### Analysis Scope: {total_employees:,} Employees ({', '.join(selected_departments)}) - Levels: {', '.join(map(str, selected_job_levels))}")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Overall Attrition Rate",
                f"{attrition_rate:.1f}%", f"{attrition_count:,} Departures")
    kpi2.metric("Overtime Risk Factor", f"{ot_diff:.1f} pts", "Higher Attrition", delta_color="inverse",
                help=f"Attrition is {ot_diff:.1f}% higher for employees working overtime ({ot_rate_yes:.1f}% vs {ot_rate_no:.1f}%)." if ot_data is not None else "Overtime data unavailable.")
    kpi3.metric("Median Income Gap", f"${income_gap:,.0f}", "Stayers Earn More", delta_color="normal" if income_gap > 0 else "inverse",
                help=f"Median income difference between stayers (${med_income_stayers:,.0f}) and leavers (${med_income_leavers:,.0f}).")
    st.markdown("--- ")

    # --- Main Analysis Tabs ---
    tab_drivers, tab_satisfaction, tab_tenure = st.tabs([
        "💰 Key Drivers (Overtime, Pay, Role)",
        "⭐ Satisfaction Impact",
        "⏳ Tenure Profile"
    ])

    # --- Key Drivers Tab ---
    with tab_drivers:
        st.subheader("Overtime, Compensation, and Job Role Influence")
        col_drv1, col_drv2 = st.columns(2)

        with col_drv1:
            st.markdown("**Overtime is a Major Factor**")
            if ot_data is not None:
                fig_ot_drv = plot_attrition_rate_bar(ot_data, x_col='OverTime', title="Attrition Rate: Overtime vs. No Overtime", height=350,
                                                     color_col='OverTime', color_map={'Yes': ATTRITION_YES_COLOR, 'No': ATTRITION_NO_COLOR})
                if fig_ot_drv:
                    fig_ot_drv.update_layout(yaxis_title='Attrition Rate (%)')
                    st.plotly_chart(fig_ot_drv, use_container_width=True)
            else:
                st.warning("Overtime data unavailable.")

        with col_drv2:
            st.markdown("**Income Distribution for Leavers vs. Stayers**")
            if 'MonthlyIncome' in df_filtered.columns:
                fig_income_viol = px.violin(df_filtered, y='MonthlyIncome', color='Attrition_Status', box=True, points=False, height=350,
                                            title=None,  # Remove title, conveyed by markdown
                                            color_discrete_map=color_map_attrition_single)
                fig_income_viol.update_layout(
                    xaxis_title=None, yaxis_title='Monthly Income ($)', showlegend=False)
                st.plotly_chart(fig_income_viol, use_container_width=True)
            else:
                st.warning("'MonthlyIncome' data unavailable.")

        st.markdown("---")
        st.markdown("**Attrition Rate by Job Roles**")
        jr_data = calculate_attrition_rate(df_filtered, 'JobRole')
        if jr_data is not None:
            jr_data_sorted = jr_data.sort_values(
                'Attrition Rate (%)', ascending=False)
            top_n = 5  # Focus on top 5
            st.markdown(f"_Highlighting Top {top_n} Highest Attrition Roles._")

            # Create Top 5 Job Role Chart directly
            fig_jr_top_drv = px.bar(jr_data_sorted.head(top_n).sort_values('Attrition Rate (%)', ascending=True),
                                    x='Attrition Rate (%)',
                                    y='JobRole',
                                    orientation='h',
                                    height=300,
                                    text='JobRole',
                                    title=None,
                                    color='Attrition Rate (%)',
                                    color_continuous_scale=SEQ_PALETTE,
                                    hover_data=['Employee Count', 'Attrition Rate (%)'])
            fig_jr_top_drv.update_layout(
                yaxis_title="Job Role",
                xaxis_title="Attrition Rate (%)",
                yaxis={'showticklabels': False},
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            fig_jr_top_drv.update_traces(
                textposition='inside', insidetextanchor='middle', textfont_size=12)
            st.plotly_chart(fig_jr_top_drv, use_container_width=True)

            # --- Deep Dive Expander ---
            top_role_names = jr_data_sorted.head(top_n)['JobRole'].tolist()
            with st.expander(f"Deep Dive: Why Attrition is High in Top {top_n} Roles ({', '.join(top_role_names)})?"):
                df_top_roles = df_filtered[df_filtered['JobRole'].isin(
                    top_role_names)]

                if not df_top_roles.empty:
                    # Calculate comparison metrics
                    comp_data = []

                    # Overtime Comparison
                    if 'OverTime' in df_filtered.columns:
                        ot_overall = df_filtered['OverTime'].value_counts(
                            normalize=True).get('Yes', 0) * 100
                        ot_top_roles = df_top_roles['OverTime'].value_counts(
                            normalize=True).get('Yes', 0) * 100
                        comp_data.append(
                            {'Metric': '% Working Overtime', 'Top 5 Roles': ot_top_roles, 'Company Average': ot_overall})

                    # Median Income Comparison
                    if 'MonthlyIncome' in df_filtered.columns:
                        income_overall = df_filtered['MonthlyIncome'].median()
                        income_top_roles = df_top_roles['MonthlyIncome'].median(
                        )
                        comp_data.append({'Metric': 'Median Monthly Income ($)',
                                         'Top 5 Roles': income_top_roles, 'Company Average': income_overall})

                    # Job Satisfaction Comparison (% Low/Medium)
                    if 'JobSatisfaction' in df_filtered.columns:  # Use original numeric column
                        js_overall_low_med = df_filtered[df_filtered['JobSatisfaction'].isin(
                            [1, 2])].shape[0] / len(df_filtered) * 100
                        js_top_roles_low_med = df_top_roles[df_top_roles['JobSatisfaction'].isin(
                            [1, 2])].shape[0] / len(df_top_roles) * 100
                        comp_data.append({'Metric': '% Low/Medium Job Satisfaction',
                                         'Top 5 Roles': js_top_roles_low_med, 'Company Average': js_overall_low_med})

                    # Environment Satisfaction Comparison (% Low/Medium)
                    if 'EnvironmentSatisfaction' in df_filtered.columns:
                        es_overall_low_med = df_filtered[df_filtered['EnvironmentSatisfaction'].isin(
                            [1, 2])].shape[0] / len(df_filtered) * 100
                        es_top_roles_low_med = df_top_roles[df_top_roles['EnvironmentSatisfaction'].isin(
                            [1, 2])].shape[0] / len(df_top_roles) * 100
                        comp_data.append({'Metric': '% Low/Medium Env. Satisfaction',
                                         'Top 5 Roles': es_top_roles_low_med, 'Company Average': es_overall_low_med})

                    if comp_data:
                        comp_df = pd.DataFrame(comp_data)
                        comp_df_melt = comp_df.melt(
                            id_vars='Metric', var_name='Group', value_name='Value')

                        fig_comp = px.bar(comp_df_melt, x='Metric', y='Value', color='Group', barmode='group',
                                          title='Key Metrics: Top 5 High-Attrition Roles vs. Company Average',
                                          color_discrete_map={
                                              'Top 5 Roles': ATTRITION_YES_COLOR, 'Company Average': ATTRITION_NO_COLOR},
                                          text_auto='.1f')  # Format text labels
                        fig_comp.update_layout(
                            yaxis_title='Percentage / Value ($)', xaxis_title='Metric', legend_title='Group')
                        fig_comp.update_traces(textposition='outside')
                        st.plotly_chart(fig_comp, use_container_width=True)

                        # Dynamic Insights based on comparison
                        st.markdown("**Summary of Differences:**")
                        insight_text = "Compared to the company average, the top 5 high-attrition roles tend to have:"
                        if ot_top_roles > ot_overall * 1.1:
                            insight_text += f"\n- **Higher Overtime:** ({ot_top_roles:.1f}% vs {ot_overall:.1f}%)"
                        if income_top_roles < income_overall * 0.9:
                            insight_text += f"\n- **Lower Median Income:** (${income_top_roles:,.0f} vs ${income_overall:,.0f})"
                        if js_top_roles_low_med > js_overall_low_med * 1.1:
                            insight_text += f"\n- **Higher proportion reporting Low/Medium Job Satisfaction:** ({js_top_roles_low_med:.1f}% vs {js_overall_low_med:.1f}%) "
                        if es_top_roles_low_med > es_overall_low_med * 1.1:
                            insight_text += f"\n- **Higher proportion reporting Low/Medium Environment Satisfaction:** ({es_top_roles_low_med:.1f}% vs {es_overall_low_med:.1f}%) "
                        st.markdown(insight_text)
                    else:
                        st.info(
                            "Not enough comparable data available for these roles.")
                else:
                    st.warning(
                        "No employees found in the identified top roles (check filters).")

            # Expander for All Job Roles Chart
            # Keep detail accessible
            with st.expander("View Attrition Rate for All Job Roles"):
                # Use original sorted data for all roles
                fig_jr_all_drv = px.bar(jr_data_sorted.sort_values('Attrition Rate (%)', ascending=True),  # Sort ascending for horizontal plot
                                        x='Attrition Rate (%)',
                                        y='JobRole',
                                        orientation='h',
                                        height=max(
                                            450, len(jr_data_sorted)*30),
                                        text='JobRole',
                                        title='Attrition Rate by Job Role (All)',
                                        color='Attrition Rate (%)',
                                        color_continuous_scale=SEQ_PALETTE,
                                        hover_data=['Employee Count', 'Attrition Rate (%)'])

                fig_jr_all_drv.update_layout(yaxis_title="", xaxis_title="Attrition Rate (%)", yaxis={
                                             'visible': False}, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10), title_x=0.5)
                fig_jr_all_drv.update_traces(
                    textposition='inside', insidetextanchor='middle', textfont_size=10)
                st.plotly_chart(fig_jr_all_drv, use_container_width=True)
        else:
            st.warning("'JobRole' data unavailable.")

        st.markdown("**Overall Driver Insights:** Overtime significantly increases attrition. Lower income and specific roles (e.g., Sales Rep, Lab Tech) are key areas of concern.")

    # --- Satisfaction Tab ---
    with tab_satisfaction:
        st.subheader("Impact of Workplace Satisfaction & Balance")
        col_sat1, col_sat2 = st.columns(2)

        with col_sat1:
            st.markdown("**Job & Environment Satisfaction**")
            satisfaction_order = ['Low', 'Medium',
                                  'High', 'Very High', 'Unknown']
            js_data = calculate_attrition_rate(
                df_filtered, 'JobSatisfaction_Label')
            es_data = calculate_attrition_rate(
                df_filtered, 'EnvironmentSatisfaction_Label')
            if js_data is not None and es_data is not None:
                js_data['Satisfaction Type'] = 'Job Satisfaction'
                es_data['Satisfaction Type'] = 'Environment Satisfaction'
                js_data = js_data.rename(
                    columns={'JobSatisfaction_Label': 'Satisfaction Level'})
                es_data = es_data.rename(
                    columns={'EnvironmentSatisfaction_Label': 'Satisfaction Level'})
                combined_sat_data = pd.concat(
                    [js_data, es_data], ignore_index=True)
                combined_sat_data['Satisfaction Level'] = pd.Categorical(
                    combined_sat_data['Satisfaction Level'], categories=satisfaction_order, ordered=True)
                combined_sat_data = combined_sat_data.sort_values(
                    ['Satisfaction Level', 'Satisfaction Type'])

                fig_sat_grouped = px.bar(combined_sat_data, x='Satisfaction Level', y='Attrition Rate (%)',
                                         color='Satisfaction Type', barmode='group',
                                         title=None,
                                         color_discrete_sequence=SEQ_PALETTE[1:4],
                                         height=350,
                                         text=combined_sat_data['Attrition Rate (%)'].apply(lambda x: f'{x:.1f}%'))
                fig_sat_grouped.update_layout(xaxis_title='Satisfaction Level', yaxis_title='Attrition Rate (%)',
                                              legend_title_text='Satisfaction Type', margin=dict(t=10, b=10))
                fig_sat_grouped.update_traces(textposition='outside')
                st.plotly_chart(fig_sat_grouped, use_container_width=True)
            else:
                st.warning(
                    "Satisfaction data (Job or Environment) unavailable.")

        with col_sat2:
            st.markdown("**Work-Life Balance**")
            if 'WorkLifeBalance_Label' in df_filtered.columns:
                wlb_data = calculate_attrition_rate(
                    df_filtered, 'WorkLifeBalance_Label')
                if wlb_data is not None:
                    wlb_order = ['Bad', 'Good', 'Better', 'Best', 'Unknown']
                    wlb_data = wlb_data.dropna(
                        subset=['Attrition Rate (%)'])  # Drop NaNs first
                    if not wlb_data.empty:
                        wlb_data['WorkLifeBalance_Label'] = pd.Categorical(
                            wlb_data['WorkLifeBalance_Label'], categories=wlb_order, ordered=True)
                        wlb_data = wlb_data.sort_values(
                            'WorkLifeBalance_Label')
                        wlb_data['TextLabel'] = wlb_data['Attrition Rate (%)'].apply(
                            lambda x: f'{x:.1f}%')

                        fig_wlb = px.bar(wlb_data,
                                         x='Attrition Rate (%)',
                                         y='WorkLifeBalance_Label',
                                         orientation='h', height=350,
                                         color='Attrition Rate (%)',
                                         color_continuous_scale=SEQ_PALETTE_R,
                                         text='TextLabel',
                                         hover_data=['Employee Count'])
                        fig_wlb.update_layout(
                            yaxis_title='Work-Life Balance',
                            xaxis_title='Attrition Rate (%)',
                            yaxis={'categoryorder': 'array',
                                   'categoryarray': wlb_order},
                            coloraxis_showscale=False,
                            margin=dict(t=10, b=10, l=100)
                        )
                        fig_wlb.update_traces(textposition='outside')
                        st.plotly_chart(fig_wlb, use_container_width=True)
                    else:
                        st.warning(
                            "No valid Work-Life Balance data to plot after filtering.")
                else:
                    st.warning(
                        "Work-Life Balance data unavailable for calculation.")
            else:
                st.warning("'WorkLifeBalance_Label' data unavailable.")

        st.markdown("**Insight:** Lower ratings in Job/Environment Satisfaction **and** poor Work-Life Balance (especially 'Bad') correlate with higher attrition.")

    # --- Tenure Tab ---
    with tab_tenure:
        st.subheader("Attrition Profile by Tenure & Promotion Cadence")
        col_ten1, col_ten2 = st.columns(2)

        with col_ten1:
            st.markdown("**Years at Company**")
            if 'YearsAtCompany' in df_filtered.columns:
                fig_tenure_viol = px.violin(df_filtered, x='Attrition_Status', y='YearsAtCompany', color='Attrition_Status',
                                            box=True, points=False, title=None,
                                            height=350,
                                            color_discrete_map=color_map_attrition_single)
                fig_tenure_viol.update_layout(
                    xaxis_title='Attrition Status', yaxis_title='Years At Company', showlegend=False, margin=dict(t=10, b=10))
                st.plotly_chart(fig_tenure_viol, use_container_width=True)
            else:
                st.warning("'YearsAtCompany' data unavailable.")

        with col_ten2:
            st.markdown("**Years Since Last Promotion**")
            if 'YearsSinceLastPromotion' in df_filtered.columns:
                bins = [-1, 1, 4, 7, df_filtered['YearsSinceLastPromotion'].max()]
                labels = ['0-1 Years', '2-4 Years', '5-7 Years', '8+ Years']
                df_filtered['PromotionGapBin'] = pd.cut(
                    df_filtered['YearsSinceLastPromotion'], bins=bins, labels=labels, right=True, ordered=True)
                promo_gap_data = calculate_attrition_rate(
                    df_filtered, 'PromotionGapBin')

                if promo_gap_data is not None:
                    promo_gap_data = promo_gap_data.dropna(
                        subset=['Attrition Rate (%)'])
                    if not promo_gap_data.empty:
                        promo_gap_data['PromotionGapBin'] = pd.Categorical(
                            promo_gap_data['PromotionGapBin'], categories=labels, ordered=True)
                        promo_gap_data = promo_gap_data.sort_values(
                            'PromotionGapBin')
                        promo_gap_data['TextLabel'] = promo_gap_data['Attrition Rate (%)'].apply(
                            lambda x: f'{x:.1f}%')

                        fig_promo_gap = px.bar(promo_gap_data,
                                               x='PromotionGapBin',
                                               y='Attrition Rate (%)',
                                               orientation='v', height=350,
                                               color='Attrition Rate (%)',
                                               color_continuous_scale=SEQ_PALETTE,
                                               text='TextLabel',
                                               hover_data=['Employee Count'])
                        fig_promo_gap.update_layout(
                            xaxis_title='Years Since Last Promotion',
                            yaxis_title='Attrition Rate (%)',
                            xaxis={'categoryorder': 'array',
                                   'categoryarray': labels},
                            coloraxis_showscale=False,
                            margin=dict(t=10, b=10)
                        )
                        fig_promo_gap.update_traces(textposition='outside')
                        st.plotly_chart(
                            fig_promo_gap, use_container_width=True)
                    else:
                        st.warning(
                            "No valid Promotion Gap data to plot after filtering.")
                else:
                    st.warning(
                        "Promotion Gap data unavailable for calculation.")
            else:
                st.warning("'YearsSinceLastPromotion' data unavailable.")

        st.markdown("**Insight:** Attrition is highest among newer employees (first few years) **and** those who haven't received a promotion in several years (e.g., 5+ years). Addressing both early career engagement and career progression paths is important.")

    st.markdown("--- ")
    st.caption("Dashboard reflects filtered data.")


# ==============================================================================
# --- Prediction Page ---
# ==============================================================================
elif page == "Predict Attrition":
    st.title(":crystal_ball: Employee Attrition Prediction")
    st.markdown(
        "Enter the details of an employee below to predict the likelihood of attrition.")

    # --- Display Prediction Result Area ---
    st.markdown("---")
    result_placeholder = st.empty()  # Placeholder for prediction results
    if st.session_state.prediction_result is not None:
        with result_placeholder.container():
            st.subheader("Prediction Result")
            if st.session_state.prediction_result == 1:
                st.warning(
                    f"**Prediction: Likely to Leave (Attrition = Yes)**")
                st.metric(label="Predicted Probability of Leaving",
                          value=f"{st.session_state.prediction_proba*100:.1f}%")
            else:
                st.success(
                    f"**Prediction: Unlikely to Leave (Attrition = No)**")
                st.metric(label="Predicted Probability of Leaving",
                          value=f"{st.session_state.prediction_proba*100:.1f}%")
            st.markdown("---")  # Add separator after showing result

    # --- Prediction Button (Placeholder) ---
    predict_button_placeholder = st.empty()  # Placeholder for the button

    if model_pipeline is None or feature_names is None:
        st.error(
            "Prediction model could not be loaded. Prediction page is unavailable.")
    else:
        # --- Define Input Fields ---
        st.subheader("Employee Details")
        col1, col2, col3 = st.columns(3)

        # Dictionary to map Feature Name -> Widget Key
        widget_keys = {}

        # Get unique values from the base dataset for dropdowns if available
        base_categories = {}
        if df_base is not None:
            for col in ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']:
                if col in df_base.columns:
                    base_categories[col] = sorted(
                        df_base[col].unique().tolist())
                else:
                    base_categories[col] = []

        # Define a helper function to create widgets and store keys
        def create_widget(widget_type, label, feature_name, options=None, **kwargs):
            key = f"pred_{feature_name.lower().replace(' ', '_')}"
            # Map feature name to its widget key
            widget_keys[feature_name] = key
            if widget_type == "number_input":
                st.number_input(label, key=key, **kwargs)
            elif widget_type == "selectbox":
                st.selectbox(label, options=options, key=key, **kwargs)
            elif widget_type == "select_slider":
                st.select_slider(label, options=options, key=key, **kwargs)
            # Add other widget types if needed

        with col1:
            st.markdown("**Demographics & Basic Info**")
            create_widget("number_input", 'Age', 'Age',
                          min_value=18, max_value=100, value=35, step=1)
            create_widget("selectbox", 'Gender', 'Gender', options=base_categories.get(
                'Gender', ['Male', 'Female']))
            create_widget("selectbox", 'Marital Status', 'MaritalStatus', options=base_categories.get(
                'MaritalStatus', ['Single', 'Married', 'Divorced']))
            create_widget("number_input", 'Distance From Home (km)',
                          'DistanceFromHome', min_value=1, max_value=30, value=5, step=1)
            create_widget("select_slider", 'Education Level', 'Education', options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: {
                          1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}.get(x))
            create_widget("selectbox", 'Education Field', 'EducationField', options=base_categories.get(
                'EducationField', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']))

        with col2:
            st.markdown("**Job & Compensation**")
            create_widget("selectbox", 'Department', 'Department', options=base_categories.get(
                'Department', ['Sales', 'Research & Development', 'Human Resources']))
            create_widget("selectbox", 'Job Role', 'JobRole', options=base_categories.get('JobRole', [
                          'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']))
            create_widget("number_input", 'Job Level', 'JobLevel',
                          min_value=1, max_value=5, value=2, step=1)
            create_widget("number_input", 'Monthly Income ($)', 'MonthlyIncome',
                          min_value=1000, max_value=20000, value=5000, step=100)
            create_widget("number_input", 'Daily Rate', 'DailyRate',
                          min_value=100, max_value=1500, value=800, step=10)
            create_widget("number_input", 'Hourly Rate ($)', 'HourlyRate',
                          min_value=30, max_value=100, value=65, step=1)
            create_widget("number_input", 'Monthly Rate', 'MonthlyRate',
                          min_value=2000, max_value=27000, value=15000, step=100)
            create_widget("number_input", 'Stock Option Level',
                          'StockOptionLevel', min_value=0, max_value=3, value=1, step=1)
            create_widget("number_input", 'Percent Salary Hike',
                          'PercentSalaryHike', min_value=10, max_value=25, value=15, step=1)

        with col3:
            st.markdown("**Experience, Satisfaction & Engagement**")
            create_widget("number_input", 'Total Working Years',
                          'TotalWorkingYears', min_value=0, max_value=40, value=10, step=1)
            create_widget("number_input", 'Number of Companies Worked At',
                          'NumCompaniesWorked', min_value=0, max_value=10, value=2, step=1)
            create_widget("number_input", 'Years At Company',
                          'YearsAtCompany', min_value=0, max_value=40, value=5, step=1)
            create_widget("number_input", 'Years In Current Role',
                          'YearsInCurrentRole', min_value=0, max_value=20, value=3, step=1)
            create_widget("number_input", 'Years Since Last Promotion',
                          'YearsSinceLastPromotion', min_value=0, max_value=15, value=1, step=1)
            create_widget("number_input", 'Years With Current Manager',
                          'YearsWithCurrManager', min_value=0, max_value=20, value=2, step=1)
            create_widget("number_input", 'Training Times Last Year',
                          'TrainingTimesLastYear', min_value=0, max_value=6, value=3, step=1)
            create_widget("selectbox", 'Works Overtime?', 'OverTime',
                          options=base_categories.get('OverTime', ['Yes', 'No']))
            create_widget("selectbox", 'Business Travel Frequency', 'BusinessTravel', options=base_categories.get(
                'BusinessTravel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']))
            create_widget("select_slider", 'Job Satisfaction', 'JobSatisfaction', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x))
            create_widget("select_slider", 'Environment Satisfaction', 'EnvironmentSatisfaction', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x))
            create_widget("select_slider", 'Job Involvement', 'JobInvolvement', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x))
            create_widget("select_slider", 'Performance Rating', 'PerformanceRating', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}.get(x))
            create_widget("select_slider", 'Relationship Satisfaction', 'RelationshipSatisfaction', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}.get(x))
            create_widget("select_slider", 'Work-Life Balance', 'WorkLifeBalance', options=[
                          1, 2, 3, 4], value=3, format_func=lambda x: {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}.get(x))

        # --- Show Button and Handle Prediction ---
        with predict_button_placeholder.container():
            st.markdown("---")  # Separator before button
            predict_button = st.button(
                # Added key to button itself
                "Predict Attrition Likelihood", type="primary", key="predict_button_main")

        if predict_button:
            # Gather data using the stored widget keys
            live_input_data = {}
            missing_feature = False
            for feature_name in feature_names:  # Iterate through features model expects
                widget_key = widget_keys.get(feature_name)
                if widget_key and widget_key in st.session_state:
                    live_input_data[feature_name] = st.session_state[widget_key]
                else:
                    st.error(
                        f"Could not find input value for feature: {feature_name}. Widget key '{widget_key}' might be missing or incorrect.")
                    missing_feature = True

            if missing_feature:
                st.session_state.prediction_result = None
                st.session_state.prediction_proba = None
                st.stop()

            # Create DataFrame from the live input data
            input_df = pd.DataFrame([live_input_data])

            # Ensure column order matches the features the pipeline expects
            try:
                # Reorder DF just in case, although iterating feature_names should maintain it
                input_df_ordered = input_df[feature_names]
            except KeyError as e:
                st.error(
                    f"DataFrame construction failed. Missing feature: {e}")
                st.error("Please check input data gathering logic.")
                st.session_state.prediction_result = None
                st.session_state.prediction_proba = None
                st.stop()
            except Exception as e:
                st.error(
                    f"An error occurred preparing DataFrame for prediction: {e}")
                st.session_state.prediction_result = None
                st.session_state.prediction_proba = None
                st.stop()

            # Make prediction
            try:
                prediction = model_pipeline.predict(input_df_ordered)[
                    0]  # Get single prediction
                probability = model_pipeline.predict_proba(input_df_ordered)[
                    0][1]  # Probability of class 1 (Attrition=Yes)

                # Store result in session state
                st.session_state.prediction_result = prediction
                st.session_state.prediction_proba = probability

                # Clear the placeholder and force rerun to display result at top
                result_placeholder.empty()
                predict_button_placeholder.empty()  # No need to clear button placeholder usually
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.session_state.prediction_result = None
                st.session_state.prediction_proba = None
