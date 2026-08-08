import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load trained model
model = tf.keras.models.load_model("ckd_model.h5")

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="CKD Prediction System", layout="centered")

st.title("Chronic Kidney Disease Prediction System")
st.write("Enter patient clinical details to predict the likelihood of CKD.")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", 1, 100)
bp = st.number_input("Blood Pressure")
sg = st.number_input("Specific Gravity")
al = st.number_input("Albumin")
su = st.number_input("Sugar")

rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
pcc = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
ba = st.selectbox("Bacteria", ["Not Present", "Present"])

bgr = st.number_input("Blood Glucose Random")
bu = st.number_input("Blood Urea")
sc = st.number_input("Serum Creatinine")
sod = st.number_input("Sodium")
pot = st.number_input("Potassium")
hemo = st.number_input("Hemoglobin")
pcv = st.number_input("Packed Cell Volume")
wbcc = st.number_input("White Blood Cell Count")
rbcc = st.number_input("Red Blood Cell Count")

htn = st.selectbox("Hypertension", ["No", "Yes"])
dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
appet = st.selectbox("Appetite", ["Good", "Poor"])
pe = st.selectbox("Pedal Edema", ["No", "Yes"])
ane = st.selectbox("Anemia", ["No", "Yes"])

# ---------------- ENCODING ---------------- #

def encode_binary(value):
    return 1 if value in ["Abnormal", "Present", "Yes", "Poor"] else 0

rbc = encode_binary(rbc)
pc = encode_binary(pc)
pcc = encode_binary(pcc)
ba = encode_binary(ba)
htn = encode_binary(htn)
dm = encode_binary(dm)
cad = encode_binary(cad)
appet = encode_binary(appet)
pe = encode_binary(pe)
ane = encode_binary(ane)

# ---------------- PREDICTION ---------------- #

if st.button("Predict"):

    # MUST MATCH TRAINING ORDER EXACTLY
    input_data = np.array([[
        age, bp, sg, al, su,
        rbc, pc, pcc, ba,
        bgr, bu, sc, sod, pot,
        hemo, pcv, wbcc, rbcc,
        htn, dm, cad, appet, pe, ane
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # CNN input (3D)
    input_cnn = input_scaled.reshape(1, input_scaled.shape[1], 1)

    # Transformer input (2D)
    input_dense = input_scaled

    # Predict
    prediction = model.predict([input_cnn, input_dense])

    # Output
    if prediction[0][0] > 0.5:
        st.error("High Risk of Chronic Kidney Disease")
    else:
        st.success("Low Risk of Chronic Kidney Disease")
