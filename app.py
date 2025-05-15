import pandas as pd
import joblib
import numpy as np
import streamlit as st


outlier_preprocessor_dropout = joblib.load('model/outlier_preprocessor_dropout.joblib')
encoders = joblib.load('model/encoder.joblib')


def preprocess_data(form_input):
    # Create a DataFrame directly from the input dictionary
    data_df = pd.DataFrame(form_input, index=[0])  # Ensure it is 2D with a single row

    # Kolom kategori
    category_cols = data_df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Lakukan encoding untuk kolom kategori
    for col in category_cols:
        encoder = encoders[col]  # Mengambil encoder yang sesuai dengan kolom
        data_df[col] = encoder.transform(data_df[col])  # Lakukan transformasi
        encoders[col] = encoder  # Simpan encoder untuk kolom tersebut

    # Columns to process for outlier handling
    columns_to_process = [
        'Admission_grade', 'Age_at_enrollment',
        'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
    ]

    # Apply outlier handler
    data_df[columns_to_process] = outlier_handler_pipeline.transform(data_df[columns_to_process])

    # Return the processed data
    return data_df
