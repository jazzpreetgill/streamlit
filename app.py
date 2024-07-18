from pycaret.regression import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('dp_insurance_charges')

input_dict = {'age' : 20, 'sex' : 'male', 'bmi' : 20, 'children' : 2, 'smoker' : 'yes', 'region' : 'southwest'}
input_df = pd.DataFrame([input_dict])
predictions_df = predict_model(estimator=model, data=input_df)
predictions = predictions_df.iloc[0]['prediction_label']
print(predictions)
