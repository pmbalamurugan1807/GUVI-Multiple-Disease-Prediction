import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ===============================
# Load datasets
# ===============================
parkinson_df = pd.read_csv(r'c:\Users\BALA\Downloads\parkinsons.csv')
kidney_df = pd.read_csv(r'c:\Users\BALA\Downloads\kidney_disease.csv')
liver_df = pd.read_csv(r'c:\Users\BALA\Downloads\indian_liver_patient.csv')

# ===============================
# Preprocessing & Model Training
# ===============================

# Parkinson
X_parkinson = parkinson_df.drop(columns=['name','status'])
y_parkinson = parkinson_df['status']
scaler_parkinson = StandardScaler()
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_parkinson, y_parkinson, test_size=0.2, random_state=42)
X_train_p_scaled = pd.DataFrame(scaler_parkinson.fit_transform(X_train_p), columns=X_train_p.columns)
X_test_p_scaled = pd.DataFrame(scaler_parkinson.transform(X_test_p), columns=X_test_p.columns)
model_parkinson = RandomForestClassifier(random_state=42)
model_parkinson.fit(X_train_p_scaled, y_train_p)

# Kidney
binary_cols_kd = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
binary_map = {'normal':1, 'abnormal':0, 'present':1, 'notpresent':0, 'yes':1, 'no':0, 'good':1, 'poor':0}
kidney_df = kidney_df.replace('?', pd.NA)
kidney_df[binary_cols_kd] = kidney_df[binary_cols_kd].apply(lambda col: col.map(binary_map))
numeric_cols_kd = kidney_df.drop(columns=['id','classification'] + binary_cols_kd).columns.tolist()
kidney_df[numeric_cols_kd] = kidney_df[numeric_cols_kd].apply(pd.to_numeric, errors='coerce')
kidney_df[numeric_cols_kd] = kidney_df[numeric_cols_kd].fillna(kidney_df[numeric_cols_kd].median())
X_kidney = kidney_df.drop(columns=['id','classification'])
y_kidney = kidney_df['classification'].map({'ckd':1,'notckd':0})
scaler_kd = StandardScaler()
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_kidney, y_kidney, test_size=0.2, random_state=42)
X_train_k_scaled = pd.DataFrame(scaler_kd.fit_transform(X_train_k), columns=X_train_k.columns)
X_test_k_scaled = pd.DataFrame(scaler_kd.transform(X_test_k), columns=X_test_k.columns)
model_kidney = RandomForestClassifier(random_state=42)
model_kidney.fit(X_train_k_scaled, y_train_k)

# Liver
liver_df['Gender'] = liver_df['Gender'].map({'Male':1,'Female':0})
X_liver = liver_df.drop(columns=['Dataset'])
y_liver = liver_df['Dataset'].map({1:1,2:0})
scaler_liver = StandardScaler()
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
X_train_l_scaled = pd.DataFrame(scaler_liver.fit_transform(X_train_l), columns=X_train_l.columns)
X_test_l_scaled = pd.DataFrame(scaler_liver.transform(X_test_l), columns=X_test_l.columns)
model_liver = RandomForestClassifier(random_state=42)
model_liver.fit(X_train_l_scaled, y_train_l)

# ===============================
# Helper Functions
# ===============================
def display_metrics(model, X_test, y_test, title):
    st.subheader(title)
    y_pred = model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.3f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.3f}")
    st.write(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

def display_summary(df, name):
    st.subheader(f" {name} Dataset Overview")

    st.write("**Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.write("**Summary Statistics:**")
    st.write(df.describe(include='all'))

    st.write("**Sample Data:**")
    st.dataframe(df.head())

    # Safe Correlation Heatmap
    st.write("**Correlation Heatmap:**")
    try:
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=['number']).copy()

        # Try to force conversion to numeric (ignore errors)
        numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

        # Drop columns with all NaN or non-numeric values
        numeric_df = numeric_df.dropna(axis=1, how='all')

        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            corr_matrix = numeric_df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")
    except Exception as e:
        st.warning(f"Could not generate heatmap due to error: {e}")

# ===============================
# Streamlit App Layout
# ===============================
st.title("Multiple Disease Prediction App")

page = st.sidebar.selectbox("Select Page", ["Prediction", "Model Metrics", "Summary"])

# ===============================
# Prediction Page
# ===============================
if page == "Prediction":
    disease = st.selectbox("Select Disease", ["Parkinson's Disease","Kidney Disease","Indian Liver Patient"])
    input_data = {}

    if disease == "Parkinson's Disease":
        for col in X_parkinson.columns:
            input_data[col] = st.number_input(col, float(X_parkinson[col].min()), float(X_parkinson[col].max()), float(X_parkinson[col].mean()))
        input_df = pd.DataFrame([input_data])
        input_scaled = pd.DataFrame(scaler_parkinson.transform(input_df), columns=input_df.columns)
        prob = model_parkinson.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob >= 0.5 else 0

    elif disease == "Kidney Disease":
        for col in X_kidney.columns:
            if col in ['rbc', 'pc']:
                input_data[col] = st.selectbox(col, options=['normal', 'abnormal'])
            elif col in ['pcc', 'ba']:
                input_data[col] = st.selectbox(col, options=['present', 'notpresent'])
            elif col in ['htn', 'dm', 'cad', 'pe', 'ane']:
                input_data[col] = st.selectbox(col, options=['yes', 'no'])
            elif col == 'appet':
                input_data[col] = st.selectbox(col, options=['good', 'poor'])
            else:
                input_data[col] = st.number_input(col, float(X_kidney[col].min()), float(X_kidney[col].max()), float(X_kidney[col].mean()))
        input_df = pd.DataFrame([input_data])
        for col in binary_cols_kd:
            input_df[col] = input_df[col].map(binary_map)
        input_scaled = pd.DataFrame(scaler_kd.transform(input_df), columns=input_df.columns)
        prob = model_kidney.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob >= 0.5 else 0

    else:
        for col in X_liver.columns:
            if col == 'Gender':
                input_data[col] = st.selectbox(col, options=['Male','Female'])
            else:
                input_data[col] = st.number_input(col, float(X_liver[col].min()), float(X_liver[col].max()), float(X_liver[col].mean()))
        input_df = pd.DataFrame([input_data])
        input_df['Gender'] = input_df['Gender'].map({'Male':1,'Female':0})
        input_scaled = pd.DataFrame(scaler_liver.transform(input_df), columns=input_df.columns)
        prob = model_liver.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob >= 0.5 else 0

    st.write(f"Prediction: {'Affected' if prediction==1 else 'Healthy'}")
    st.write(f"Probability of being affected: {prob:.3f}")

# ===============================
# Model Metrics Page
# ===============================
elif page == "Model Metrics":
    st.header("Model Evaluation Metrics")
    display_metrics(model_parkinson, X_test_p_scaled, y_test_p, "Parkinson's Disease")
    display_metrics(model_kidney, X_test_k_scaled, y_test_k, "Kidney Disease")
    display_metrics(model_liver, X_test_l_scaled, y_test_l, "Indian Liver Patient")

# ===============================
# Summary Page
# ===============================
elif page == "Summary":
    st.header("Dataset Summary")    
    dataset_choice = st.selectbox("Select Dataset", ["Parkinson's Disease", "Kidney Disease", "Indian Liver Patient"])

    if dataset_choice == "Parkinson's Disease":
        display_summary(parkinson_df, "Parkinson's Disease")
    elif dataset_choice == "Kidney Disease":
        display_summary(kidney_df, "Kidney Disease")
    else:
        display_summary(liver_df, "Indian Liver Patient")
