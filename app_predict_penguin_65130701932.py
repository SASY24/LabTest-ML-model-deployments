

import pickle
import pandas as pd
import streamlit as st

# โหลดโมเดลและตัวแปลงที่จำเป็น
with open('model_penguin_65130701932.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# สร้างฟังก์ชันสำหรับทำนายผล
def predict_species(island, culmen_length, culmen_depth, flipper_length, body_mass, sex):
    # สร้าง DataFrame สำหรับข้อมูลใหม่
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length],
        'culmen_depth_mm': [culmen_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'sex': [sex]
    })
    
    # แปลงค่าข้อมูลใหม่
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])
    
    # ทำนายผล
    y_pred_new = model.predict(x_new)
    result = species_encoder.inverse_transform(y_pred_new) 
    
    return result[0]

# ตั้งค่าชื่อหัวข้อของแอป
st.title("Penguin Species Prediction")

# รับข้อมูลจากผู้ใช้
island = st.selectbox("Select Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, max_value=100.0, value=37.0)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, max_value=100.0, value=19.3)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0, max_value=300.0, value=192.3)
body_mass = st.number_input("Body Mass (g)", min_value=0, max_value=10000, value=3750)
sex = st.selectbox("Select Sex", ['MALE', 'FEMALE'])

# เมื่อผู้ใช้กดปุ่ม "Predict"
if st.button("Predict"):
    # ทำนายผล
    predicted_species = predict_species(island, culmen_length, culmen_depth, flipper_length, body_mass, sex)
    
    # แสดงผลลัพธ์
    st.write(f'Predicted Species: {predicted_species}')


