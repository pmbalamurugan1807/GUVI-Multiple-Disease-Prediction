# GUVI-Multiple-Disease-Prediction

Streamlit project to predict multiple diseases, in this app its Parkinson's, Kidney disease and Indian liver disease

(Note: entire project is done in 1 python file(multiplediseaseprediction.py))

step 1: This involves processing of data(indian_liver_patient.csv, kidney_disease.csv and parkinsons.csv) into a format thats more easier to model for predictions which is mainly binary encoding and drop columns with only 1 value(aka those that dont contribute to the prediction analysis at all ) and also modelling the data for predictions with randomforestclassifier on a 80/20 split( for training and testing).

Step 2: Preparing a function to display sklearn metrics and confusion matrix to show how credible the model is based given data
<img width="500" height="281" alt="Screenshot (21)" src="https://github.com/user-attachments/assets/4b069ee0-ba1d-4ec4-b083-d059fc539beb" />

Step 3: Preparing the streamlit page itself with including the navigation sidebar and the disease prediction pages and each disease can be selected with the dropdown selectbox as shown below and disease prediction is dynamically shown at the bottom of the page
<img width="500" height="281" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/6d3ae6fd-2a0d-4a00-a6dc-f84c07183239" />
<img width="500" height="281" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/fac52156-2d55-4b67-b036-13cfa14d4f49" />
