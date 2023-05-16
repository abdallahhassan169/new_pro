import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib

DATA_PATH = r"alldata.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1)

disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
	"Disease": disease_counts.index,
	"Counts": disease_counts.values
})

# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:,1:]
y = data.iloc[:,0]

# Cleaning data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
ImputedX = imputer.fit(X)
X = ImputedX.transform(X)
symptoms = data[1:].columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}
ImputedModule=SimpleImputer(missing_values=np.nan, strategy='most_frequent', fill_value=None, verbose=0, copy=True)
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
#Data division
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 8)
model = RandomForestClassifier(random_state=60)

model.fit(X, y)
preds = model.predict(X_test)

joblib.dump(model , 'my_model00.joblib',compress=3)