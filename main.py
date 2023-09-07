import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# cleaning the data

df = pd.read_csv("./diabetes.csv") # having_diabetes = 1 , not_having_diabetes = 0
df["gender"] = (df["gender"] == "male").astype(int)
df["diabetes"] = (df["diabetes"] == "Diabetes").astype(int)  # male = 1 , female = 0
df.drop(["patient_number"], axis=1, inplace=True)
df["bmi"] = df["bmi"].str.replace(",", ".").astype(float)
df["waist_hip_ratio"] = df["waist_hip_ratio"].str.replace(",", ".").astype(float)
df["chol_hdl_ratio"] = df["chol_hdl_ratio"].str.replace(",", ".").astype(float)
df_class_true = df[df["diabetes"] == 1]
df_class_false = df[df["diabetes"] == 0]
df_class_false = df_class_false.sample(df_class_true["diabetes"].count(), random_state=42)
df = pd.concat([df_class_false, df_class_true], axis=0)
trainset, testset = train_test_split(df, test_size=0.25, random_state=42)
X_train = trainset.drop(["diabetes"], axis=1)
y_train = trainset["diabetes"]
X_test = testset.drop(["diabetes"], axis=1)
y_test = testset["diabetes"]


# traning the model

error_rate = []
for i in range(1, 89):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

req_k_value = error_rate.index(min(error_rate))+1

classifier = KNeighborsClassifier(n_neighbors=req_k_value)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)*100


# the web page


st.title("Predicting Diabetes")
st.write("### Please Enter your data")

gender = st.selectbox("Select Gender", ("male", "female"))
st.write("gender: ", gender)

age = st.number_input("Age", min_value=1, value=25, step=1)
st.write('The current number is ', age)

height = st.number_input("Height in cm", min_value=1, value=175, step=1) 
st.write('The current number is ', height)

weight = st.number_input("Weight in kg", min_value=1, value=80, step=1) 
st.write('The current number is ', weight)

cholesterol = st.number_input("Your Cholesterol Level in mg/dl", value=170)
st.write('The current number is ', cholesterol)

glucose = st.number_input("Your Glucose Level in mg/dl", value=125)
st.write('The current number is ', glucose)

hdl_chol = st.number_input("Your HDL Level in mg/dl", min_value=1, value=40)
st.write('The current number is ', hdl_chol)

systolic_bp = st.number_input("Systolic blood pressure in mmHg", value=110)
st.write('The current number is ', systolic_bp)

diastolic_bp = st.number_input("Diastolic blood pressure in mmHg", value=70)
st.write('The current number is ', diastolic_bp)

waist = st.number_input("Waist circumference in cm", min_value=1, value=80, step=1) 
st.write('The current number is ', waist)

hip = st.number_input("Hip circumference in cm", min_value=1, value=80) 
st.write('The current number is ', hip)

bmi = (703 * weight * 2.20462) / ((height / 2.54)**2)
chol_hdl_ratio = cholesterol / hdl_chol
waist_hip_ratio = waist / hip
gender_num = 1 if gender=="male" else 0 

button = st.button("Predict")


if (button):
    my_data = {
        "cholesterol": cholesterol,
        "glucose": glucose,
        "hdl_chol": hdl_chol,
        "chol_hdl_ratio": chol_hdl_ratio,
        "age": age,
        "gender": gender_num,
        "height": height / 2.54,
        "weight": weight * 2.20462,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "waist": waist / 2.54,
        "hip": hip / 2.54,
        "waist_hip_ratio": waist_hip_ratio
    }
    my_df = pd.DataFrame(data=my_data, index=[0])
    prediction = classifier.predict(my_df)[0]
    if (prediction):
        st.write("#### You have diabetes")
    else:
        st.write("#### You do not have diabetes")

    st.write(f"##### prediction accuracy: {round(accuracy, 2)}%")
