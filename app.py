import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Water_ Jet')

# Step 1: Load the Model
model = open('rf.pickle', 'rb')
pipeline = pickle.load(model)
model.close()

# Step 2: Create a UI for the front-end user
water_pressure = st.number_input('water pressure (Mpa)', min_value=193, max_value=234)
SOD = st.number_input('SOD(mm)', min_value=3, max_value=7)
Traverse_rate = st.number_input('Traverse rate (mm/min)', min_value=15, max_value=35)

# Step 3: Change User Input to Model Input data
data = {'water_pressure': [water_pressure], 'SOD': [SOD], 'Traverse_rate': [Traverse_rate]}
input_data = pd.DataFrame(data)

# Step 4: Get Predictions and Print the result
if st.button('Predict'):
    result = pipeline.predict(input_data)
    st.table(input_data)
    st.success(str(result[0]))
