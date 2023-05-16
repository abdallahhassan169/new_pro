import numpy as np
import pandas as pd
from scipy.stats import mode
import joblib
from sklearn.preprocessing import LabelEncoder
from specialization import * 

from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS


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


model = joblib.load('my_model00.joblib')
app = Flask(__name__)
CORS(app)
@app.route('/result',methods=['POST','GET'])
def result():
    symptoms = request.form['symptoms']
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * 1167
    for symptom in symptoms:
        if symptom in data_dict['symptom_index'].keys(): 
    	    index = data_dict["symptom_index"][symptom]
    	    input_data[index] = 1
        else:
             return "your symptom not found"
    	
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    key =model.predict(input_data)[0]
    # making final prediction by taking mode of all predictions
    
    pred = data_dict["predictions_classes"][key]
    if pred in women:
         res = {"disease": pred , "specialization":"Gynecology specialty"}
    elif pred in dental:
        res = {"disease": pred , "specialization":"dental"}
    elif pred in nose:
        res = {"disease": pred , "specialization":"Ear, nose and throat specialty"}    
    elif pred in inter or pred in Internal_diseases_specialty:
        res = {"disease": pred , "specialization":"Internal diseases specialty"}    
    elif pred in dermatology:
        res = {"disease": pred , "specialization":"dermatology"}   
    elif pred in Respiratory_or_chest_diseases:
          res = {"disease": pred , "specialization":"Respiratory_or_chest_diseases"}
    elif pred in Orthopedic :
         res = {"disease": pred , "specialization":"Orthopedic"}
    elif pred in Department_of_Brain_and_Nerves:
         res = {"disease": pred , "sp-ecialization":"Department_of_Brain_and_Nerves"}
    elif pred in Department_of_Urology:
         res = {"disease": pred , "specialization":"Department_of_Urology"}
    elif pred in Department_of_Psychiatry_and_Neurology:
         res = {"disease": pred , "specialization":"Department_of_Psychiatry_and_Neurology"}
    elif pred in Department_of_Hematology:
         res = {"disease": pred , "specialization":"Department_of_Hematology"}
    elif pred in Ophthalmology:
         res = {"disease": pred , "specialization":"Ophthalmology"}


         
        

 

    return jsonify(str(res))
if __name__ == '__main__':
    app.run(debug=False)


