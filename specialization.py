import pandas as pd 
dental = pd.read_csv('dental binary.csv').dropna(axis = 1).values
nose = pd.read_csv('ENT BINARY TEST OK.csv').dropna(axis = 1).values
women= pd.read_csv('النسا.csv').dropna(axis = 1).values
inter= pd.read_csv('interor.csv', encoding='ISO-8859-1')
Internal_diseases_specialty =['Diabetes', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease', 'AIDS', 'Gastroenteritis', 'Hypertension', 'Migraine','Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria','Typhoid', 'Hepatitis A', 'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Heart attack','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthritis','Arthritis','Chronic cholestasis','Peptic ulcer disease','Gastroenteritis','Jaundice','Hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis.','Hypertension','Hypertensive disease','Coronary arteriosclerosis','Coronary heart disease','Congestive heart failure','Cerebrovascular accident','Myocardial infarction','Hypercholesterolemia','Cardiomyopathy','Mitral valve insufficiency','Aortic valve stenosis','Pulmonary hypertension','Tricuspid valve insufficiency','Pericardial effusion','Hypothyroidism','Hyperthyroidism','Hyperglycemia','Hypoglycemia','Diabetic ketoacidosis','Gastroesophageal reflux disease','Peptic ulcer disease','Colitis','Diverticulitis','Diverticulosis','Cholecystitis','Hiatal hernia','Hemorrhoids','Gastritis','Bacteremia','Anemia','Sickle cell anemia','Thrombocytopaenia','Pancytopenia','Neutropenia','Hepatitis C','Hepatitis B','Candidiasis','Oral candidiasis','HIV','HIV infections','Gout','Peripheral vascular disease', 'Tachycardia sinus','Transient ischemic attack','Pulmonary embolism','Ischemia','Deep vein thrombosis','Embolism pulmonary','Thrombus']
dermatology = ['Acne','Psoriasis','Impetigo','Allergy','Cellulitis','Decubitus ulcer']
Respiratory_or_chest_diseases =['Bronchial Asthma','Tuberculosis','Pneumonia','Asthma','Pulmonary embolism', 'Pulmonary edema', 'Chronic obstructive airway disease','Upper respiratory infection','Emphysema','Bronchitis','Bronchial spasm','Pneumonia aspiration','Influenza','Pneumocystis carinii pneumonia','Respiratory failure','Spasm bronchial']
Orthopedic =['Cervical spondylosis', 'Osteoarthritis','Arthritis','Degenerative polyarthritis','Arthritis','Osteoporosis','Fibroid tumor','Osteomyelitis']
Department_of_Brain_and_Nerves=['Migraine','Paralysis (brain hemorrhage)','Paroxysmal Positional Vertigo','Confusion','Epilepsy',"Parkinson's", 'disease', 'Hemiparesis','Aphasia','Tonc-clonic epilepsy','Tonc-clonic seizures']
Department_of_Urology =['Urinary tract infection','Insufficiency renal','Chronic kidney failure','Pyelonephritis']
Department_of_Psychiatry_and_Neurology =['Depression','Depressive disorder','Dementia','Anxiety state','Psychotic disorder','Bipolar disorder','Paranoia','Schizophrenia','Personality disorder','Manic disorder','Delirium','Delusion','Affect labile','Suicide attempt','Dependence']
Department_of_Oncology =['Malignant neoplasms','Primary malignant neoplasm','Malignant tumor of colon','Carcinoma colon','Malignant neoplasm of prostate','Carcinoma prostate','Malignant neoplasm of breast','Carcinoma breast','Malignant neoplasm of lung','Carcinoma lung','Lymphoma','Neoplasm metastasis','Primary carcinoma of the liver cells','Melanoma']
Department_of_Hematology=['Septicemia','Systemic infection','Sepsis'] 
Ophthalmology=['Glaucoma',]      
