import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open(r'C:/Users/Abhay Patil/Desktop/Project/MDPS/saved models/diabetes_model_ready.sav', 'rb'))
heart_disease_model = pickle.load(open(r'C:/Users/Abhay Patil/Desktop/Project/MDPS/saved models/heart_disease_model_ready.sav', 'rb'))
parkinsons_model = pickle.load(open(r'C:/Users/Abhay Patil/Desktop/Project/MDPS/saved models/parkinsons_model_ready.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('personal health guardian',
                          ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                          icons=['activity','heart','person'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    if 'show_placeholder' not in st.session_state:
        st.session_state.show_placeholder = True

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies e.g. 6 , 1', value='' if st.session_state.show_placeholder else '')
    with col2:
        Glucose = st.text_input('Glucose Level e.g. 148 , 85', value='' if st.session_state.show_placeholder else '')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value e.g. 72 , 66', value='' if st.session_state.show_placeholder else '')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value e.g. 35 , 29', value='' if st.session_state.show_placeholder else '')
    with col2:
        Insulin = st.text_input('Insulin Level e.g. 0 , 0', value='' if st.session_state.show_placeholder else '')
    with col3:
        BMI = st.text_input('BMI value e.g 33.6 , 26.6', value='' if st.session_state.show_placeholder else '')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value e.g. 0.627 , 0.351', value='' if st.session_state.show_placeholder else '')
    with col2:
        Age = st.text_input('Age of the Person 50 , 31', value='' if st.session_state.show_placeholder else '')

    if any([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
        st.session_state.show_placeholder = False

    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            input_data = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
            diab_prediction = diabetes_model.predict([input_data])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            st.error("⚠️ Please fill in all input fields with valid numeric values.")

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age eg 63 , 67')
    with col2:
        sex = st.text_input('Sex (0 = Female, 1 = Male)')
    with col3:
        cp = st.text_input('Chest Pain types 3 , 0')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure 145 , 160')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl 233 , 286')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl 1 , 0')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results 0 , 0')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved 150 , 108')
    with col3:
        exang = st.text_input('Exercise Induced Angina 0 , 1')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise 2.3 , 1.5')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment 0 , 1')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy 0 , 3')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            input_data = [float(i) for i in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            heart_prediction = heart_disease_model.predict([input_data])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
        except ValueError:
            st.error("⚠️ Please fill in all input fields with valid numeric values.")

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz) eg 119.992 , 197.076')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz) eg 157.302 , 206.896')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz) eg 74.997 , 192.055')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%) eg  0.00784 , 0.00289')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs) eg 0.00007 , 0.00001')
    with col1:
        RAP = st.text_input('MDVP:RAP eg 0.0037 , 0.00166')
    with col2:
        PPQ = st.text_input('MDVP:PPQ eg 0.00554 , 0.00168')
    with col3:
        DDP = st.text_input('Jitter:DDP eg 0.01109 , 0.00498')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer eg 0.04374 , 0.01098')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB) eg 0.426 , 0.097')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3 eg 0.02182 , 0.00563')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5 eg 0.0313 , 0.0068')
    with col3:
        APQ = st.text_input('MDVP:APQ eg 0.02971 , 0.00802')
    with col4:
        DDA = st.text_input('Shimmer:DDA eg 0.06545 , 0.01689')
    with col5:
        NHR = st.text_input('NHR eg 0.02211 , 0.00339')
    with col1:
        HNR = st.text_input('HNR eg 21.033 , 26.775')
    with col2:
        RPDE = st.text_input('RPDE eg 0.414783 , 0.422229')
    with col3:
        DFA = st.text_input('DFA eg 0.815285 , 0.741367')
    with col4:
        spread1 = st.text_input('spread1 eg -4.813031 , -7.3483')
    with col5:
        spread2 = st.text_input('spread2 eg 0.266482 , 0.177551')
    with col1:
        D2 = st.text_input('D2 eg 2.301442 , 1.743867')
    with col2:
        PPE = st.text_input('PPE eg 0.284654 , 0.085569')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            input_data = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer,
                                             Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA,
                                             spread1, spread2, D2, PPE]]
            parkinsons_prediction = parkinsons_model.predict([input_data])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            st.error("⚠️ Please fill in all input fields with valid numeric values.")

    st.success(parkinsons_diagnosis)
