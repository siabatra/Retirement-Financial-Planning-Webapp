import pandas as pd
import numpy as np
import joblib
import os
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import zscore

# Load dataset
file_path = "C:/Users/kriti/Downloads/data_final.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file at {file_path} was not found.")

df = pd.read_csv(file_path)
print("File loaded successfully!")

# Verify dataset integrity
print("Dataset Columns:", df.columns.tolist())
print("Data Types:\n", df.dtypes)
print("First Few Rows:\n", df.head())

# Required features
required_features = ['Total_Invested_Amount', 'Expected_ROI', 'Retirement_Age', 'Age', 'Inflation_Rate',
                     'Savings_Rate', 'Cost_of_Living']
missing_features = [feat for feat in required_features if feat not in df.columns]

if missing_features:
    print(f"Warning: Missing features in dataset: {missing_features}")
    exit()

# Handle missing values
df = df.copy()
df.fillna({
    'Total_Invested_Amount': df['Total_Invested_Amount'].median(),
    'Expected_ROI': df['Expected_ROI'].mean(),
    'Retirement_Age': 65,
    'Age': df['Age'].median(),
    'Inflation_Rate': 0.025,
    'Savings_Rate': df['Savings_Rate'].median(),
    'Cost_of_Living': df['Cost_of_Living'].median()
}, inplace=True)

# Ensure valid values
df['Expected_ROI'] = df['Expected_ROI'].clip(0, 0.2)
df = df[df['Retirement_Age'] > df['Age']]

# Outlier detection using IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

for col in ['Total_Invested_Amount', 'Savings_Rate', 'Cost_of_Living']:
    df = remove_outliers_iqr(df, col)

# Feature Engineering
df['Years_to_Retirement'] = df['Retirement_Age'] - df['Age']
df['Real_Return_Rate'] = df['Expected_ROI'] - df['Inflation_Rate']
df['Annual_Investment_Growth'] = df['Total_Invested_Amount'] * df['Expected_ROI']
df['Future_Investment_Value'] = df['Total_Invested_Amount'] * ((1 + df['Real_Return_Rate']).clip(lower=0.01) ** df['Years_to_Retirement'])
df['Savings_Years_Interaction'] = df['Savings_Rate'] * df['Years_to_Retirement']

# Feature selection
features = ['Total_Invested_Amount', 'Expected_ROI', 'Retirement_Age', 'Age', 'Inflation_Rate',
            'Savings_Rate', 'Cost_of_Living', 'Years_to_Retirement', 'Real_Return_Rate',
            'Annual_Investment_Growth', 'Savings_Years_Interaction']
target = 'Future_Investment_Value'

X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
joblib.dump(scaler, 'scaler.pkl')

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Manually Define Optimized Parameters
best_params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200
}

# Train final model
best_xgb_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
best_xgb_model.fit(X_train, y_train)

# Save model
joblib.dump(best_xgb_model, 'investment_model.pkl')

# Save model as formatted JSON
model_json = best_xgb_model.get_booster().save_config()
formatted_json = json.loads(model_json)
with open("investment_model.json", "w") as json_file:
    json.dump(formatted_json, json_file, indent=4)

# Evaluate model
y_pred = best_xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')

"""
# Output:
File loaded successfully!
Dataset Columns: ['Age', 'Education_Level', 'Occupation', 'Number_of_Dependents', 'Location', 'Work_Experience', 'Marital_Status', 'Employment_Status', 'Household_Size', 'Homeownership_Status', 'Type_of_Housing', 'Gender', 'Primary_Mode_of_Transportation', 'Income', 'Career_Level', 'Financial_Status', 'Housing', 'Food', 'Transportation', 'Education', 'Total_Invested_Amount', 'Emergency_Funds', 'Tax_Rate', 'Cost_of_Living', 'Monthly_Savings', 'Income_Growth_Rate', 'Budget_Adjustments', 'Expected_ROI', 'Healthcare_Cost', 'Debt', 'Savings_Rate', 'Desired_Expenses', 'Inflation_Rate', 'Retirement_Age', 'Life_Expectancy']
Data Types:
 Age                               float64
Education_Level                    object
Occupation                         object
Number_of_Dependents              float64
Location                           object
Work_Experience                   float64
Marital_Status                     object
Employment_Status                  object
Household_Size                    float64
Homeownership_Status               object
Type_of_Housing                    object
Gender                             object
Primary_Mode_of_Transportation     object
Income                            float64
Career_Level                       object
Financial_Status                   object
Housing                           float64
Food                              float64
Transportation                    float64
Education                         float64
Total_Invested_Amount             float64
Emergency_Funds                   float64
Tax_Rate                          float64
Cost_of_Living                    float64
Monthly_Savings                   float64
Income_Growth_Rate                float64
Budget_Adjustments                float64
Expected_ROI                      float64
Healthcare_Cost                   float64
Debt                              float64
Savings_Rate                      float64
Desired_Expenses                  float64
Inflation_Rate                    float64
Retirement_Age                    float64
Life_Expectancy                   float64
dtype: object
First Few Rows:
     Age Education_Level  Occupation  Number_of_Dependents Location  ...  Savings_Rate Desired_Expenses Inflation_Rate  Retirement_Age Life_Expectancy
0  56.0        Master's  Technology                   5.0    Urban  ...      28457.33         24527.93           2.35            63.0            84.0
1  69.0     High School     Finance                   0.0    Urban  ...      29700.21         44334.32           2.35            63.0            82.0
2  46.0      Bachelor's  Technology                   1.0    Urban  ...      22181.83         33326.02           2.35            63.0            83.0
3  32.0     High School      Others                   2.0    Urban  ...      11428.21         42327.27           2.35            63.0            83.0
4  60.0      Bachelor's     Finance                   3.0    Urban  ...     232062.93        221683.44           2.35            63.0            83.0

[5 rows x 35 columns]
Mean Squared Error: 6478775.2344
# Indicates low variance in errors, meaning the predictions are quite close to the actual values

Mean Absolute Error: 373.7872 
# On average, the models predictions are off by around $373. This is a very low error ie. model's predictions are highly accurate

R^2 Score: 0.9985
# 99.85% of the variance in Future_Investment_Value is explained by the model ie. model has extremely high accuracy
"""