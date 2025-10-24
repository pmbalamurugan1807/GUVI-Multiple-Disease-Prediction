# GUVI-Multiple-Disease-Prediction

Streamlit project to predict multiple diseases, in this app its Parkinson's, Kidney disease and Indian liver disease

(Note: entire project is done in 1 python file(multiplediseaseprediction.py))

step 1: This involves processing of data(indian_liver_patient.csv, kidney_disease.csv and parkinsons.csv) into a format thats more easier to model for predictions which is mainly binary encoding and drop columns with only 1 value(aka those that dont contribute to the prediction analysis at all ) and also modelling the data for predictions with randomforestclassifier on a 80/20 split( for training and testing).

Step 2: Preparing a function to display sklearn metrics and confusion matrix to show how credible the model is based given data and another function for the summary of given dataset 

<img width="500" height="281" alt="Screenshot (68)" src="https://github.com/user-attachments/assets/a04e1807-1b90-4792-9679-9bdac6564b83" />

Step 3: Preparing the streamlit page itself with including the navigation sidebar and the disease prediction pages and each disease can be selected with the dropdown selectbox as shown below and a selectbox to pick a model for prediction and the final disease prediction for the model is dynamically shown at the bottom of the page

<img width="500" height="281" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/109232e2-7088-4e26-afea-326860915703" />
<img width="500" height="281" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/78cb1659-ca8f-472b-9017-1ef39863db9c" />
<img width="500" height="281" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/f6f05111-7382-48a3-bee1-b1962f8d0878" />


Step 4: Model evaluation metric section where we can see the metrics for each model and each disease shown here 

<img width="500" height="281" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/f980d975-8008-4a67-91c1-efc192674d14" />


Step 4: And finally summary tab which shows different dataset summary and the corelation heatmap for them 

<img width="500" height="281" alt="Screenshot (55)" src="https://github.com/user-attachments/assets/4a609d63-3526-4b0f-9efe-e53051c593ab" />
