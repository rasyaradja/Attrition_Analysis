# Employee Attrition Analysis & Prediction Dashboard

## Introduction

This project addresses the challenge of high employee attrition faced by the HR department at Jaya Jaya Maju. The primary goal is to analyze historical employee data to identify key factors driving attrition and to develop a predictive model to assess the likelihood of current employees leaving. The findings and predictive tool are presented through an interactive web dashboard built with Streamlit.

## Data

*   **Source:** The analysis is based on a cleaned employee dataset containing various demographic, job-related, compensation, and satisfaction metrics from [Here](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee).
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

The [Streamlit application](https://attritionrate-rasya.streamlit.app/) provides two main pages:

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

*   Clone the repository: `git clone https://github.com/rasyaradja/Attrition_Analysis`
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

*   **Prerequisites:**
    *   A GitHub account.
    *   Your project code (including `submission/dashboard.py`, `submission/requirements.txt`, `submission/model/logistic_regression_pipeline.joblib`, `.gitignore` but **excluding** `.streamlit/secrets.toml`) pushed to a GitHub repository.
    *   A Supabase account and project with the `cleaned_employee_data` table populated and RLS configured to allow reads for the `anon` role.
*   **Steps:**
    1.  Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
    2.  Click "**New app**" and choose "**From existing repo**".
    3.  Select the GitHub repository and the branch containing your latest code.
    4.  Set the "**Main file path**" to `submission/dashboard.py`.
    5.  Click "**Advanced settings...**".
    6.  Navigate to the "**Secrets**" section.
    7.  Copy the *entire contents* of your **local** `submission/.streamlit/secrets.toml` file (containing your actual Supabase URL and Key) and paste it into the secrets text box.
    8.  Click "**Save**" for the secrets.
    9.  Click "**Deploy!**". Streamlit will build the environment and deploy your application.
*   [Link to current deployed version](https://attritionrate-rasya.streamlit.app/)

## Creator

*   **Name:** Rasya Radja
*   **LinkedIn:** [https://www.linkedin.com/in/rasyaradja/](https://www.linkedin.com/in/rasyaradja/) 