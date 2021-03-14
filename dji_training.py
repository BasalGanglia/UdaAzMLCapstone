# The following code is adapted from Microsoft tutorials here:
#https://microsoftlearning.github.io/mslearn-dp100/instructions/11-tune-hyperparameters.html
# Import libraries
import argparse, joblib, os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Input dataset
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

# Hyperparameters
parser.add_argument('--min_samples_split', type=int, dest='min_samples_split', default=5, help='min_samples_split')
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')
parser.add_argument('--min_samples_leaf', type=int, dest='min_samples_leaf', default=5, help='min_samples_leaf')
parser.add_argument('--max_depth', type=int, dest='max_depth', default=100, help='max depth')
# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
run.log('min_samples_split',  np.int(args.min_samples_split))
run.log('n_estimators',  np.int(args.n_estimators))
run.log('min_samples_leaf',  np.float(args.min_samples_leaf))
run.log('max_depth',  np.int(args.max_depth))
# load the diabetes dataset
print("Loading Data...")
dji_df = run.input_datasets['training_data'].to_pandas_dataframe() # Get the training data from the estimator input

# Separate features and labels
#X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values
y = dji_df['y']
X = dji_df.drop(['y'], axis=1)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train a Random Forest classification model with the specified hyperparameters
print('Training a classification model')
model = RandomForestClassifier(min_samples_leaf=args.min_samples_leaf, min_samples_split=args.min_samples_split, max_depth = args.max_depth,
                                   n_estimators=args.n_estimators).fit(X_train, y_train)
# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
#print(f'can this be seen? the auc is {auc}')
print('AUC: ' + str(auc))
run.log("AUC", np.float(auc))

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/dji_model.pkl')

run.complete()