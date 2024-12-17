import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Load the data
link = "D:\\Datasets\\telecom_customer_details.csv"
df=pd.read_csv(link)

#Display basic information about the data set
print(df.info())

# Generate summary statistics
print(df.describe())

#Visualize the distribution of key variables
plt.figure(figsize=(12,6))
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Distribution of monthly charges")
plt.show()

# Check for co-relations between variables
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
# correlation_matrix=df.corr()
plt.figure(figsize=(12,6))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Pre-processing and feature engineering

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify numeric and categorical columns

numeric_features = ['Tenure (months)','MonthlyCharges','TotalCharges']
categorical_features = ['Gender','InternetService','Contract Type']

# Create preprocessing steps for numeric feature

numeric_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='constant',fill_value='missing')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,categorical_features)
    ]
)

# Fit and transform the data

# Features
X=df.drop('Churn',axis=1)

# Target variables
y=df['Churn']
X_processed = preprocessor.fit_transform(X)

# Feature engineering

df['charge_tenure_ratio']=df['TotalCharges']/df['Tenure (months)']
df['multiple_services'] = (df['InternetService'] !='No') & (df['PhoneService']=='Yes')

# Model selection and training

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Initialize model

models = {
    'Logistic Regression':LogisticRegression(),
    'Random Forest':RandomForestClassifier(),
    'XGBoost':XGBClassifier()
}

# Perform cross validation for each model

for name, model in models.items():
    scores = cross_val_score(model,X_processed, y, cv=5, scoring='accuracy')
    print(f"{name} - Mean accuracy : {scores.mean():.3f}(+/-{scores.std()*2:.3f})")

# Model evaluation and interpretation

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap

# Split the data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Generate classification report
print(classification_report(y_test,y_pred))

# Create confusion matrix

print(confusion_matrix(y_test,y_pred))

# Feature importance

feature_importance = model.feature_importances_
feature_names = preprocessor.get_feature_names_out()

# Sort features by importance
sorted_idx = feature_importance.argsort()
for idx in sorted_idx:
    print(f"{feature_names[idx]}:{feature_importance[idx]:.4f}")

# Shap values for model interpretaion

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Model deployment and monitoring

from flask import Flask,request,jsonify
import joblib
import os
app = Flask(__name__)
directory = r'C:\Users\praka\PycharmProjects\pythonProject'
filename = 'preprocessor.joblib'
filepath = os.path.join(directory, filename)
 # Ensure the directory exists
if not os.path.exists(directory): os.makedirs(directory)
# Save the trained model to the specified directory
joblib.dump(model, filepath)

# Load trained model and preprocessor
model=joblib.load('churn_model.joblib')
preprocessor=joblib.load('preprocessor.joblib')
@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    processed_data = preprocessor.transform(pd.DataFrame(data,index=[0]))
    prediction = model.predict(processed_data)
    return jsonify({'churn_prediction':int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
