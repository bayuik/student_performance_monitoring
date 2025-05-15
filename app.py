import pandas as pd
import joblib
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# Set page configuration
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 15px;
        border-bottom: 2px solid #E5E7EB;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 15px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid #E5E7EB;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-top: 20px;
    }
    .graduate {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #166534;
    }
    .dropout {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #991B1B;
    }
    .info-text {
        font-size: 14px;
        color: #6B7280;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Definisi class OutlierHandler
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for col in self.cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - 1.5 * IQR
            self.upper_bounds[col] = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        X = X.copy()  # Menghindari perubahan pada X asli
        for col in self.cols:
            lower = self.lower_bounds[col]
            upper = self.upper_bounds[col]
            X[col] = np.clip(X[col], lower, upper)  # Mengatasi outlier
        return X

# Load model dan encoder pada awal aplikasi
@st.cache_resource
def load_models():
    outlier_handler = joblib.load('model/outlier_preprocessor_dropout.joblib')
    encoder = joblib.load('model/encoder.joblib')
    model = joblib.load('model/rf_classifier_dropout.joblib')
    return outlier_handler, encoder, model

outlier_handler_pipeline, encoders, model = load_models()

def preprocess_data(form_input):
    # Create a DataFrame directly from the input dictionary
    data_df = pd.DataFrame(form_input, index=[0])

    # Kolom kategori
    category_cols = data_df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Lakukan encoding untuk kolom kategori
    for col in category_cols:
        encoder = encoders[col]  # Mengambil encoder yang sesuai dengan kolom
        data_df[col] = encoder.transform(data_df[col])  # Lakukan transformasi

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

# Fungsi untuk melakukan prediksi
def prediction(data):
    # Memproses data input menggunakan preprocessing
    data_processed = preprocess_data(data)

    # Melakukan prediksi menggunakan model
    prediction = model.predict(data_processed)

    # Mengembalikan hasil prediksi (0: Dropout, 1: Graduate)
    return prediction

# Function to create default values dictionary
def get_default_values():
    return {
        'marital_status': 'single',
        'application_mode': 'International student',
        'application_order': 1,
        'course': 'Tourism',
        'daytime_evening_attendance': 'daytime',
        'previous_qualification': 'Secondary education',
        'previous_qualification_grade': 160,
        'nationality': 'Portuguese',
        'mothers_qualification': 'Secondary Education - 12th Year of Schooling or Eq.',
        'fathers_qualification': 'Higher Education - Degree',
        'mothers_occupation': 'Intermediate Technicians and Professionals',
        'fathers_occupation': 'Intermediate Technicians and Professionals',
        'admission_grade': 142.5,
        'displaced': 'yes',
        'educational_special_needs': 'no',
        'debtor': 'no',
        'tuition_fees_up_to_date': 'no',
        'gender': 'male',
        'scholarship_holder': 'no',
        'age_at_enrollment': 19,
        'international': 'no',
        'curricular_units_1st_sem_enrolled': 6,
        'curricular_units_1st_sem_evaluations': 6,
        'curricular_units_1st_sem_approved': 6,
        'curricular_units_1st_sem_grade': 14,
        'curricular_units_2nd_sem_enrolled': 6,
        'curricular_units_2nd_sem_evaluations': 6,
        'curricular_units_2nd_sem_approved': 6,
        'curricular_units_2nd_sem_grade': 13.66666667,
        'unemployment_rate': 13.9,
        'inflation_rate': -0.3,
        'gdp': 0.79
    }

# Option lists
def get_option_lists():
    return {
        'marital_status_options': ['single', 'married', 'widower', 'divorced', 'facto union', 'legally separated'],
        'application_mode_options': ['1st phase - general contingent', 'Ordinance No. 612/93', '1st phase - special contingent (Azores Island)',
                                    'Holders of other higher courses', 'Ordinance No. 854-B/99', 'International student',
                                    '1st phase - special contingent (Madeira Island)', '2nd phase - general contingent',
                                    '3rd phase - general contingent', 'Ordinance No. 533-A/99, item b2 (Different Plan)',
                                    'Ordinance No. 533-A/99, item b3 (Other Institution)', 'Over 23 years old', 'Transfer',
                                    'Change of course', 'Technological specialization diploma holders',
                                    'Change of institution/course', 'Short cycle diploma holders',
                                    'Change of institution/course (International)'],
        'course_options': ['Biofuel Production Technologies', 'Animation and Multimedia Design',
                          'Social Service (evening attendance)', 'Agronomy', 'Communication Design',
                          'Veterinary Nursing', 'Informatics Engineering', 'Equinculture', 'Management',
                          'Social Service', 'Tourism', 'Nursing', 'Oral Hygiene',
                          'Advertising and Marketing Management', 'Journalism and Communication',
                          'Basic Education', 'Management (evening attendance)'],
        'daytime_evening_options': ['daytime', 'evening'],
        'previous_qualification_options': ['Secondary education', 'Higher education - bachelor\'s degree',
                                          'Higher education - degree', 'Higher education - master\'s',
                                          'Higher education - doctorate', 'Frequency of higher education',
                                          '12th year of schooling - not completed', '11th year of schooling - not completed',
                                          'Other - 11th year of schooling', '10th year of schooling',
                                          '10th year of schooling - not completed',
                                          'Basic education 3rd cycle (9th/10th/11th year) or equiv.',
                                          'Basic education 2nd cycle (6th/7th/8th year) or equiv.',
                                          'Technological specialization course', 'Higher education - degree (1st cycle)',
                                          'Professional higher technical course', 'Higher education - master (2nd cycle)'],
        'nationality_options': ['Portuguese', 'German', 'Spanish', 'Italian', 'Dutch', 'English', 'Lithuanian',
                               'Angolan', 'Cape Verdean', 'Guinean', 'Mozambican', 'Santomean', 'Turkish',
                               'Brazilian', 'Romanian', 'Moldova (Republic of)', 'Mexican', 'Ukrainian',
                               'Russian', 'Cuban', 'Colombian'],
        'qualification_options': ['Secondary Education - 12th Year of Schooling or Eq.', 'Higher Education - Bachelor\'s Degree',
                                 'Higher Education - Degree', 'Higher Education - Master\'s', 'Higher Education - Doctorate',
                                 'Frequency of Higher Education', '12th Year of Schooling - Not Completed',
                                 '11th Year of Schooling - Not Completed', '7th Year (Old)', 'Other - 11th Year of Schooling',
                                 '10th Year of Schooling', 'General commerce course',
                                 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.', 'Technical-professional course',
                                 '7th year of schooling', '2nd cycle of the general high school course',
                                 '9th Year of Schooling - Not Completed', '8th year of schooling', 'Unknown',
                                 'Can\'t read or write', 'Can read without having a 4th year of schooling',
                                 'Basic education 1st cycle (4th/5th year) or equiv.',
                                 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
                                 'Technological specialization course', 'Higher education - degree (1st cycle)',
                                 'Specialized higher studies course', 'Professional higher technical course',
                                 'Higher Education - Master (2nd cycle)', 'Higher Education - Doctorate (3rd cycle)'],
        'occupation_options': ['Student', 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
                               'Specialists in Intellectual and Scientific Activities', 'Intermediate Technicians and Professionals',
                               'Administrative staff', 'Personal Services, Security and Safety Workers and Sellers',
                               'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
                               'Skilled Workers in Industry, Construction and Craftsmen',
                               'Installation and Machine Operators and Assembly Workers', 'Unskilled Workers',
                               'Armed Forces Professions', 'Other Situation', 'blank', 'Health professionals',
                               'teachers', 'Specialists in information and communication technologies (ICT)',
                               'Intermediate level science and engineering technicians and professions',
                               'Technicians and professionals, of intermediate level of health',
                               'Intermediate level technicians from legal, social, sports, cultural and similar services',
                               'Office workers, secretaries in general and data processing operators',
                               'Data, accounting, statistical, financial services and registry-related operators',
                               'Other administrative support staff', 'Personal service workers', 'Sellers',
                               'Personal care workers and the like', 'Skilled construction workers and the like, except electricians',
                               'Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
                               'Workers in food processing, woodworking, clothing and other industries and crafts',
                               'Cleaning workers', 'Unskilled workers in agriculture, animal production, fisheries and forestry',
                               'Unskilled workers in extractive industry, construction, manufacturing and transport',
                               'Meal preparation assistants'],
        'yes_no_options': ['yes', 'no'],
        'gender_options': ['male', 'female']
    }

# Main application
st.markdown("<h1 class='main-header'>🎓 Prediksi Dropout Mahasiswa</h1>", unsafe_allow_html=True)

# Brief explanation
st.markdown("""
<p>Aplikasi ini membantu memprediksi apakah seorang mahasiswa berisiko dropout berdasarkan berbagai faktor akademik dan personal.</p>
""", unsafe_allow_html=True)

# Get default values and options
defaults = get_default_values()
options = get_option_lists()

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Data Personal", "Data Pendidikan", "Data Akademik", "Data Ekonomi"])

with st.form(key='input_form'):
    # TAB 1: PERSONAL DATA
    with tab1:
        st.markdown("<div class='section-header'>Data Personal</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox('Gender', options['gender_options'],
                                index=options['gender_options'].index(defaults['gender']))
            age_at_enrollment = st.number_input('Age at Enrollment', min_value=15, max_value=100,
                                              value=defaults['age_at_enrollment'])
            marital_status = st.selectbox('Marital Status', options['marital_status_options'],
                                        index=options['marital_status_options'].index(defaults['marital_status']))
            nationality = st.selectbox('Nationality', options['nationality_options'],
                                     index=options['nationality_options'].index(defaults['nationality']))

        with col2:
            international = st.selectbox('International Student', options['yes_no_options'],
                                       index=options['yes_no_options'].index(defaults['international']))
            displaced = st.selectbox('Displaced (Living away from home)', options['yes_no_options'],
                                   index=options['yes_no_options'].index(defaults['displaced']))
            educational_special_needs = st.selectbox('Educational Special Needs', options['yes_no_options'],
                                                  index=options['yes_no_options'].index(defaults['educational_special_needs']))
            scholarship_holder = st.selectbox('Scholarship Holder', options['yes_no_options'],
                                           index=options['yes_no_options'].index(defaults['scholarship_holder']))

    # TAB 2: EDUCATIONAL BACKGROUND
    with tab2:
        st.markdown("<div class='section-header'>Latar Belakang Pendidikan</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            previous_qualification = st.selectbox('Previous Qualification', options['previous_qualification_options'],
                                                index=options['previous_qualification_options'].index(defaults['previous_qualification']))
            previous_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0, max_value=200,
                                                         value=defaults['previous_qualification_grade'])
            mothers_qualification = st.selectbox('Mother\'s Qualification', options['qualification_options'],
                                               index=options['qualification_options'].index(defaults['mothers_qualification']))
            mothers_occupation = st.selectbox('Mother\'s Occupation', options['occupation_options'],
                                            index=options['occupation_options'].index(defaults['mothers_occupation']))

        with col2:
            application_mode = st.selectbox('Application Mode', options['application_mode_options'],
                                          index=options['application_mode_options'].index(defaults['application_mode']))
            application_order = st.number_input('Application Order', min_value=0, max_value=9,
                                              value=defaults['application_order'])
            fathers_qualification = st.selectbox('Father\'s Qualification', options['qualification_options'],
                                               index=options['qualification_options'].index(defaults['fathers_qualification']))
            fathers_occupation = st.selectbox('Father\'s Occupation', options['occupation_options'],
                                            index=options['occupation_options'].index(defaults['fathers_occupation']))

    # TAB 3: ACADEMIC DATA
    with tab3:
        st.markdown("<div class='section-header'>Data Akademik</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            course = st.selectbox('Course', options['course_options'],
                                index=options['course_options'].index(defaults['course']))
            daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', options['daytime_evening_options'],
                                                    index=options['daytime_evening_options'].index(defaults['daytime_evening_attendance']))
            admission_grade = st.number_input('Admission Grade', min_value=0.0, max_value=200.0,
                                            value=float(defaults['admission_grade']), step=0.1)

            st.markdown("<div class='section-header'>Semester 1</div>", unsafe_allow_html=True)
            curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem Enrolled', min_value=0.0, max_value=26.0,
                                                            value=float(defaults['curricular_units_1st_sem_enrolled']))
            curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem Evaluations', min_value=0.0, max_value=45.0,
                                                                value=float(defaults['curricular_units_1st_sem_evaluations']))
            curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', min_value=0.0, max_value=26.0,
                                                              value=float(defaults['curricular_units_1st_sem_approved']))
            curricular_units_1st_sem_grade = st.number_input('Curricular Units 1st Sem Grade', min_value=0.0, max_value=20.0,
                                                           value=float(defaults['curricular_units_1st_sem_grade']), step=0.1)

        with col2:
            debtor = st.selectbox('Debtor', options['yes_no_options'],
                                index=options['yes_no_options'].index(defaults['debtor']))
            tuition_fees_up_to_date = st.selectbox('Tuition Fees Up To Date', options['yes_no_options'],
                                                index=options['yes_no_options'].index(defaults['tuition_fees_up_to_date']))

            st.markdown("<div style='height: 56px;'></div>", unsafe_allow_html=True)  # Spacer for alignment

            st.markdown("<div class='section-header'>Semester 2</div>", unsafe_allow_html=True)
            curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem Enrolled', min_value=0.0, max_value=23.0,
                                                             value=float(defaults['curricular_units_2nd_sem_enrolled']))
            curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem Evaluations', min_value=0.0, max_value=33.0,
                                                                value=float(defaults['curricular_units_2nd_sem_evaluations']))
            curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', min_value=0.0, max_value=20.0,
                                                              value=float(defaults['curricular_units_2nd_sem_approved']))
            curricular_units_2nd_sem_grade = st.number_input('Curricular Units 2nd Sem Grade', min_value=0.0, max_value=20.0,
                                                           value=float(defaults['curricular_units_2nd_sem_grade']), step=0.1)

    # TAB 4: ECONOMIC FACTORS
    with tab4:
        st.markdown("<div class='section-header'>Faktor Ekonomi</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            unemployment_rate = st.number_input('Unemployment Rate (%)', min_value=0.0, max_value=100.0,
                                              value=defaults['unemployment_rate'], step=0.1)
        with col2:
            inflation_rate = st.number_input('Inflation Rate (%)', value=float(defaults['inflation_rate']), step=0.1)
        with col3:
            gdp = st.number_input('GDP (Billion)', value=float(defaults['gdp']), step=0.01)

    # Submit button - centered and prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.form_submit_button(label='PREDIKSI DROPOUT')

# Show results outside the form
if submit_button:
    with st.spinner('Memproses prediksi...'):
        # Prepare data for prediction
        input_data = {
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': daytime_evening_attendance,
            'Previous_qualification': previous_qualification,
            'Previous_qualification_grade': previous_qualification_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qualification,
            'Fathers_qualification': fathers_qualification,
            'Mothers_occupation': mothers_occupation,
            'Fathers_occupation': fathers_occupation,
            'Admission_grade': admission_grade,
            'Age_at_enrollment': age_at_enrollment,
            'Displaced': displaced,
            'Educational_special_needs': educational_special_needs,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_fees_up_to_date,
            'Gender': gender,
            'Scholarship_holder': scholarship_holder,
            'International': international,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
            'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
            'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
            'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
            'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
            'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp
        }

        # Create a DataFrame for prediction
        input_df = pd.DataFrame(input_data, index=[0])

        # Get prediction
        result = prediction(input_df)

        # Display result with better styling
        st.markdown("<h2 style='text-align: center; margin-top: 30px;'>Hasil Prediksi</h2>", unsafe_allow_html=True)

        if result[0] == 0:
            st.markdown("<div class='result-box dropout'>Prediksi: Mahasiswa DROPOUT</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-text'>
                Berdasarkan data yang dimasukkan, model memprediksi bahwa mahasiswa ini berisiko tinggi untuk dropout.
                Perhatikan faktor-faktor akademik dan personal untuk intervensi dini.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box graduate'>Prediksi: Mahasiswa GRADUATE</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-text'>
                Berdasarkan data yang dimasukkan, model memprediksi bahwa mahasiswa ini kemungkinan akan menyelesaikan
                pendidikan dengan baik.
            </div>
            """, unsafe_allow_html=True)
else:
    # Show welcome message when app first loads
    st.info("""
    👋 Selamat datang di aplikasi prediksi dropout mahasiswa.

    Aplikasi ini menggunakan model Machine Learning untuk memprediksi kemungkinan seorang mahasiswa dropout berdasarkan
    berbagai faktor akademik, personal, dan sosial-ekonomi.

    Silakan isi semua informasi di tab-tab di atas dan klik tombol 'PREDIKSI DROPOUT' untuk mendapatkan hasil prediksi.
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #E5E7EB;">
    <p style="color: #6B7280; font-size: 12px;">
        Sistem Prediksi Dropout Mahasiswa | Dibuat dengan Streamlit dan Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)