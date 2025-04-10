# Employee Attrition Analysis & Prediction Dashboard

## Introduction

This project addresses the challenge of high employee attrition faced by the HR department at Jaya Jaya Maju. The primary goal is to analyze historical employee data to identify key factors driving attrition and to develop a predictive model to assess the likelihood of current employees leaving. The findings and predictive tool are presented through an interactive web dashboard built with Streamlit.

## Data

*   **Source:** The analysis is based on a cleaned employee dataset containing various demographic, job-related, compensation, and satisfaction metrics.
*   **Connection:** The deployed Streamlit dashboard connects directly to a **Supabase** database table (`cleaned_employee_data`) to fetch the latest data for analysis and visualization. This ensures the dashboard reflects the most current information available in the database.
*   **Key Features Used:** Age, OverTime, MonthlyIncome, JobRole, JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, YearsAtCompany, YearsSinceLastPromotion, Department, JobLevel, BusinessTravel, DistanceFromHome, etc.

## Methodology

1.  **Exploratory Data Analysis (EDA):**
    *   Calculated overall attrition rates and rates segmented by various factors.
    *   Identified key drivers through comparative analysis:
        *   Employees working **Overtime** show significantly higher attrition.
        *   Lower **Monthly Income** correlates with higher attrition.
        *   Specific **Job Roles** (e.g., Sales Representative, Laboratory Technician) exhibit higher rates.
        *   Lower **Job Satisfaction**, **Environment Satisfaction**, and **Work-Life Balance** ratings are associated with increased attrition.
        *   Attrition is higher among **newer employees** and those with longer **times since last promotion**.
    *   Visualizations were created using Plotly for clarity and interactivity within the dashboard.

2.  **Predictive Modeling:**
    *   A **Logistic Regression** model was trained using `scikit-learn`.
    *   A preprocessing pipeline (handling scaling for numerical features and one-hot encoding for categorical features) was included using `scikit-learn`'s `Pipeline` and saved using `joblib`.
    *   The model predicts the probability of an employee leaving (Attrition = Yes).

## Dashboard Features

The Streamlit application provides two main pages:

1.  **Dashboard Analysis:**
    *   **Executive KPIs:** Displays Overall Attrition Rate, Overtime Risk Factor (difference in attrition rate), and Median Income Gap between leavers and stayers.
    *   **Filtering:** Allows users to filter the analysis by Department and Job Level.
    *   **Analysis Tabs:**
        *   *Key Drivers:* Visualizes the impact of Overtime, compares Income distributions, shows Attrition Rate by Top 5 Job Roles (with names inside bars), and includes a "Deep Dive" expander comparing key metrics for these top roles against the company average.
        *   *Satisfaction Impact:* Shows grouped bar charts comparing Job Satisfaction and Environment Satisfaction, alongside a bar chart for Work-Life Balance impact.
        *   *Tenure Profile:* Displays a violin plot for Years at Company distribution and a bar chart showing attrition rate based on Years Since Last Promotion bins.

2.  **Predict Attrition:**
    *   Provides an intuitive form to input details for a single employee across various features.
    *   Uses the loaded `joblib` model pipeline to predict the likelihood of attrition.
    *   Displays the prediction result (Likely/Unlikely to Leave) and the calculated probability.

## Technology Stack

*   **Language:** Python 3
*   **Data Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Dashboard:** Streamlit
*   **Visualization:** Plotly Express
*   **Database:** Supabase (PostgreSQL)
*   **Serialization:** Joblib

## Setup & Deployment

**1. Local Setup:**

*   Clone the repository: `git clone <repository_url>`
*   Navigate to the project directory.
*   Install required packages: `pip install -r submission/requirements.txt`
*   **Configure Supabase Credentials:**
    *   Create a directory `submission/.streamlit/`.
    *   Inside that directory, create a file named `secrets.toml`.
    *   Add your Supabase URL and Key (obtain from Supabase project settings > API):
        ```toml
        # submission/.streamlit/secrets.toml
        SUPABASE_URL = "YOUR_SUPABASE_PROJECT_URL"
        SUPABASE_KEY = "YOUR_SUPABASE_ANON_PUBLIC_KEY"
        ```
    *   **Important:** Ensure `.streamlit/secrets.toml` is added to your `.gitignore` file to prevent committing secrets.
*   Run the dashboard: `streamlit run submission/dashboard.py`

**2. Deployment (Streamlit Community Cloud):**

* 

## Creator

*   **Name:** Rasya Radja
*   **LinkedIn:** [https://www.linkedin.com/in/rasyaradja/](https://www.linkedin.com/in/rasyaradja/) 