import pandas as pd
import joblib
import os
import warnings

# Ignore warnings for cleaner output (optional)
warnings.filterwarnings('ignore')

# --- Configuration ---
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the model file relative to the script's location
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'logistic_regression_pipeline.joblib')

# --- Load Model ---
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"Model pipeline loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure 'logistic_regression_pipeline.joblib' is in the 'model' subdirectory.")
    exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# --- Prediction Function ---
def predict_attrition(employee_data):
    """
    Predicts attrition for new employee data.

    Args:
        employee_data (pd.DataFrame): A DataFrame containing employee data
                                       with columns matching the training data
                                       (before preprocessing, excluding 'Attrition').

    Returns:
        np.ndarray: An array of predictions (0 for No Attrition, 1 for Yes Attrition).
        np.ndarray: An array of probabilities for Attrition=1.
    """
    if not isinstance(employee_data, pd.DataFrame):
        raise ValueError("Input employee_data must be a pandas DataFrame.")

    # Ensure columns match the order expected by the preprocessor/model
    # (Fetching required columns from the model's preprocessor step if possible)
    try:
        # Access the preprocessor from the pipeline
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Get numerical and categorical feature names the preprocessor was trained on
        num_features = [t[2][i] for t in preprocessor.transformers_ if t[0] == 'num' for i in range(len(t[2]))]
        cat_features = [t[2][i] for t in preprocessor.transformers_ if t[0] == 'cat' for i in range(len(t[2]))]
        required_columns = num_features + cat_features
        
        # Check if all required columns are in the input DataFrame
        missing_cols = set(required_columns) - set(employee_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in input data: {missing_cols}")

        # Reorder input DataFrame columns to match training order
        employee_data_ordered = employee_data[required_columns]

    except AttributeError:
        print("Warning: Could not automatically determine required feature order from pipeline.")
        print("Assuming input DataFrame columns are already in the correct order.")
        employee_data_ordered = employee_data
    except Exception as e:
         print(f"Warning: Error getting feature names from pipeline: {e}")
         print("Assuming input DataFrame columns are already in the correct order.")
         employee_data_ordered = employee_data


    try:
        # Predict using the full pipeline (handles preprocessing)
        predictions = model_pipeline.predict(employee_data_ordered)
        probabilities = model_pipeline.predict_proba(employee_data_ordered)[:, 1] # Probability of class 1
        return predictions, probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    # Create sample employee data (replace with actual new data)
    # Ensure column names match the original dataset EXACTLY
    # Use realistic values based on EDA/data description
    sample_data = {
        'Age': [35, 22, 48],
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
        'DailyRate': [800, 500, 1200],
        'Department': ['Research & Development', 'Sales', 'Human Resources'],
        'DistanceFromHome': [5, 20, 2],
        'Education': [3, 2, 4], # 1 'Below College', 2 'College', 3 'Bachelor', 4 'Master', 5 'Doctor'
        'EducationField': ['Life Sciences', 'Marketing', 'Human Resources'],
        'EnvironmentSatisfaction': [4, 1, 3], # 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
        'Gender': ['Male', 'Female', 'Male'],
        'HourlyRate': [70, 45, 90],
        'JobInvolvement': [3, 2, 4], # 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
        'JobLevel': [2, 1, 4],
        'JobRole': ['Laboratory Technician', 'Sales Representative', 'Manager'],
        'JobSatisfaction': [4, 2, 3], # 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
        'MaritalStatus': ['Married', 'Single', 'Divorced'],
        'MonthlyIncome': [5000, 2500, 15000],
        'MonthlyRate': [15000, 8000, 25000],
        'NumCompaniesWorked': [2, 0, 5],
        'OverTime': ['No', 'Yes', 'No'],
        'PercentSalaryHike': [15, 12, 20],
        'PerformanceRating': [3, 3, 4], # 1 'Low', 2 'Good', 3 'Excellent', 4 'Outstanding'
        'RelationshipSatisfaction': [3, 1, 4], # 1 'Low', 2 'Medium', 3 'High', 4 'Very High'
        'StockOptionLevel': [1, 0, 2], # 0 to 3
        'TotalWorkingYears': [10, 1, 25],
        'TrainingTimesLastYear': [3, 2, 3],
        'WorkLifeBalance': [3, 2, 4], # 1 'Bad', 2 'Good', 3 'Better', 4 'Best'
        'YearsAtCompany': [5, 1, 20],
        'YearsInCurrentRole': [3, 0, 15],
        'YearsSinceLastPromotion': [1, 0, 4],
        'YearsWithCurrManager': [2, 0, 10]
        # Excluded: EmployeeId, Attrition, EmployeeCount, StandardHours, Over18
    }
    
    new_employees = pd.DataFrame(sample_data)

    print("\n--- Predicting Sample Data ---")
    predictions, probabilities = predict_attrition(new_employees)

    if predictions is not None:
        result_df = pd.DataFrame({
            'Prediction (0=No, 1=Yes)': predictions,
            'Probability of Attrition (%)': [f"{p*100:.2f}%" for p in probabilities]
        })
        # Optionally add some original data for context
        result_df = pd.concat([new_employees[['Age', 'JobRole', 'MonthlyIncome']].reset_index(drop=True), result_df], axis=1)
        
        print(result_df)
