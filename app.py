import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np

# Charger le modèle de régression logistique
model_uri = 'runs:/1a7169fbbd6846d8ab50ae7171fc7529/logistic_regression_model'
model = mlflow.sklearn.load_model(model_uri)

# Interface utilisateur
st.title("Prédiction de défaut de crédit")

credit_lines = st.slider("Nombre de lignes de crédit", 0, 10)
loan_amt = st.slider("Montant du prêt", 1000, 50000)
total_debt = st.slider("Dette totale", 5000, 100000)
income = st.slider("Revenu", 20000, 150000)
years_employed = st.slider("Années d'emploi", 0, 40)
fico_score = st.slider("Score FICO", 300, 850)

if st.button("Prédire"):
    input_data = np.array([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]])
    prediction = model.predict(input_data)
    st.write(f"Prédiction : {'Défaut' if prediction[0] == 1 else 'Pas de défaut'}")