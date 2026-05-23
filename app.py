import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

st.title("Predicción de Rotación de Personal")
st.write("Modelo predictivo para estimar si un empleado puede renunciar.")
st.success("VERSIÓN NUEVA FUNCIONANDO")

df = pd.read_csv("employee_churn_dataset.csv")

if "Employee ID" in df.columns:
    df = df.drop(columns=["Employee ID"])

# Convertir todas las columnas tipo texto a números
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

st.write("Tipos de datos después de codificar:")
st.write(df.dtypes)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

st.subheader("Prueba de predicción")

empleado = st.selectbox("Selecciona un empleado de prueba", range(len(X_test)))

nuevo_empleado = X_test.iloc[[empleado]]

if st.button("Predecir rotación"):
    prediccion = model.predict(nuevo_empleado)

    if prediccion[0] == 1:
        st.error("Resultado: El empleado tiene probabilidad de renunciar.")
    else:
        st.success("Resultado: El empleado no presenta alta probabilidad de renuncia.")