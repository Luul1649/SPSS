import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# --- Title ---
st.title("Student Performance Prediction System")
st.write("Predict a student's final grade (G3) using multiple algorithms")

# --- Load Models ---
lr_model = pickle.load(open("linear_model.pkl", "rb"))
dt_model = pickle.load(open("decision_tree_model.pkl", "rb"))
rf_model = pickle.load(open("random_forest_model.pkl", "rb"))
opt_rf_model = pickle.load(open("optimized_rf_model.pkl", "rb"))
nn_model = tf.keras.models.load_model("neural_network_model_clean.h5",compile=False)

# --- Sidebar Inputs ---
st.sidebar.header("Enter Student Details")

# Numeric Inputs
age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Weekly Study Time (1-4)", 1, 4, 2)
failures = st.sidebar.slider("Past Class Failures", 0, 4, 0)
absences = st.sidebar.slider("Number of Absences", 0, 50, 5)
G1 = st.sidebar.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.sidebar.slider("Second Period Grade (G2)", 0, 20, 10)
famrel = st.sidebar.slider("Family Relationship (1-5)", 1, 5, 4)
freetime = st.sidebar.slider("Free Time After School (1-5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out With Friends (1-5)", 1, 5, 3)
Dalc = st.sidebar.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
Walc = st.sidebar.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 2)
health = st.sidebar.slider("Current Health (1-5)", 1, 5, 4)

# Categorical Inputs (0/1 encoding)
def cat_input(label):
    return st.sidebar.selectbox(label, [0,1])

sex = cat_input("Sex (0=F, 1=M)")
address = cat_input("Address (0=Urban,1=Rural)")
famsize = cat_input("Family Size (0=LE3,1=GT3)")
Pstatus = cat_input("Parent Status (0=Together,1=Apart)")
schoolsup = cat_input("Extra School Support (0=No,1=Yes)")
famsup = cat_input("Family Support (0=No,1=Yes)")
paid = cat_input("Extra Paid Classes (0=No,1=Yes)")
activities = cat_input("Extra Activities (0=No,1=Yes)")
nursery = cat_input("Attended Nursery (0=No,1=Yes)")
higher = cat_input("Wants Higher Education (0=No,1=Yes)")
internet = cat_input("Internet Access at Home (0=No,1=Yes)")
romantic = cat_input("Romantic Relationship (0=No,1=Yes)")

# --- Prepare Features ---
# For simplicity, set other features like Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime to defaults
features = np.array([[
    0,          # school (default)
    sex,
    age,
    address,
    famsize,
    Pstatus,
    2,          # Medu default
    2,          # Fedu default
    0,          # Mjob default
    0,          # Fjob default
    0,          # reason default
    0,          # guardian default
    1,          # traveltime default
    studytime,
    failures,
    schoolsup,
    famsup,
    paid,
    activities,
    nursery,
    higher,
    internet,
    romantic,
    famrel,
    freetime,
    goout,
    Dalc,
    Walc,
    health,
    absences,
    G1,
    G2
]])

# --- Select Model ---
model_choice = st.selectbox(
    "Choose Algorithm",
    ["Linear Regression", "Decision Tree", "Random Forest", "Optimized Random Forest", "Neural Network"]
)

# --- Predict ---
if st.button("Predict Final Grade"):
    if model_choice == "Linear Regression":
        pred = lr_model.predict(features)
    elif model_choice == "Decision Tree":
        pred = dt_model.predict(features)
    elif model_choice == "Random Forest":
        pred = rf_model.predict(features)
    elif model_choice == "Optimized Random Forest":
        pred = opt_rf_model.predict(features)
    else:
        pred = nn_model.predict(features).flatten()

    st.success(f"Predicted Final Grade ({model_choice}): {pred[0]:.2f}")

# --- Optional Visualization ---
if st.checkbox("Show Study Time vs Final Grade Plot"):
    df_plot = pd.read_csv("student-mat.csv", sep=";")
    plt.figure(figsize=(8,5))
    sns.boxplot(x="studytime", y="G3", data=df_plot)
    st.pyplot(plt)