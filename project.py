import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as stc
import time
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Descriptive Analytics ", page_icon="🌏", layout="wide")

theme_plotly = None

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

df = pd.read_csv('loan-train.csv')

st.title("LOAN PREDICTION APP")
st.image("loan_image.jpg")

st.sidebar.header("Please Filter Here:")
Education = st.sidebar.multiselect(
    "Select the Education:",
    options=df["Education"].unique(),
    default=df["Education"].unique()
)
Property_Area = st.sidebar.multiselect(
    "Select the Property_Area:",
    options=df["Property_Area"].unique(),
    default=df["Property_Area"].unique()
)
Gender = st.sidebar.multiselect(
    "Select the Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

df_selection = df.query(
    "Education == @Education & Property_Area == @Property_Area & Gender == @Gender"
)

# Home page function
def HomePage():
    # 1. Print dataframe
    with st.expander("🌎 My database"):
        shwdata = st.multiselect('Filter:', df_selection.columns, default=["Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status"])
        st.dataframe(df_selection[shwdata], use_container_width=True)

    # 2. Compute top Analytics
    total_ApplicantIncome = df_selection['ApplicantIncome'].sum()
    ApplicantIncome_mode = df_selection['ApplicantIncome'].mode()
    ApplicantIncome_mean = df_selection['ApplicantIncome'].mean()
    ApplicantIncome_median = df_selection['ApplicantIncome'].median()

    # 3. Display metrics
    total1, total2, total3, total4, total5 = st.columns(5, gap='large')
    with total1:
        st.info('Total Applicant Income', icon="🔍")
        st.metric(label='sum TZS', value=f"{total_ApplicantIncome:}")

    with total2:
        st.info('Most frequently', icon="🔍")
        st.metric(label='Mode TZS', value=f"{ApplicantIncome_mode[0]:}")

    with total3:
        st.info('Applicant Income Average', icon="🔍")
        st.metric(label='Mean TZS', value=f"{ApplicantIncome_mean:}")

    with total4:
        st.info('Applicant Income Margin', icon="🔍")
        st.metric(label='Median TZS', value=f"{ApplicantIncome_median:}")

    st.markdown("""---""")


# Graphs function
def Graphs():
    total_ApplicantIncome = int(df_selection["ApplicantIncome"].sum())
    average_ApplicantIncome = round(df_selection["ApplicantIncome"].mean(), 2)

    # 1. Simple bar graph
    ApplicantIncome_by_Property_Area = df_selection.groupby(by=["Property_Area"]).count()[["ApplicantIncome"]].sort_values(by="ApplicantIncome")
    fig_ApplicantIncome = px.bar(
        ApplicantIncome_by_Property_Area,
        x="ApplicantIncome",
        y=ApplicantIncome_by_Property_Area.index,
        orientation="h",
        title="ApplicantIncome_by_Property_Area",
        color_discrete_sequence=["#0083B8"] * len(ApplicantIncome_by_Property_Area),
        template="plotly_white",
    )

    fig_ApplicantIncome.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # 2. Simple line graph
    ApplicantIncome_by_Dependents = df_selection.groupby(by=["Dependents"]).count()[["ApplicantIncome"]]
    fig_Dependents = px.line(
        ApplicantIncome_by_Dependents,
        x=ApplicantIncome_by_Dependents.index,
        orientation="v",
        y="ApplicantIncome",
        title="ApplicantIncome_by_Dependents",
        color_discrete_sequence=["#0083B8"] * len(ApplicantIncome_by_Dependents),
        template="plotly_white",
    )
    fig_Dependents.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    left_column, right_column, center = st.columns(3)
    left_column.plotly_chart(fig_Dependents, use_container_width=True)
    right_column.plotly_chart(fig_ApplicantIncome, use_container_width=True)

    # Pie chart
    with center:
        fig = px.pie(df_selection, values='LoanAmount', names='Property_Area', title='Property_Area by LoanAmount')
        fig.update_layout(legend_title="Property_Area", legend_y=0.9)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

    st.markdown('Applicant Income VS Loan Amount ')    
    st.bar_chart(df[['ApplicantIncome','LoanAmount']].head(20))

# Progress bar function
def ProgressBar():
    st.markdown("""<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #99ff99 , #FFFF00)}</style>""", unsafe_allow_html=True,)
    target = 3000000000
    current = df_selection['ApplicantIncome'].sum()
    percent = round((current / target * 100))
    my_bar = st.progress(0)

    if percent > 100:
        st.subheader("Target 100 completed")
    else:
        st.write("you have ", percent, " % ", " of ", (format(target, ',d')), " TZS")
        for percent_complete in range(percent):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text="Target percentage")

# Data preprocessing and model training
def preprocess_and_train_model():
    # Drop non-numeric columns
    df_clean = df.drop(['Loan_ID'], axis=1)

    # Fill missing values
    df_clean['Gender'].fillna(df_clean['Gender'].mode()[0], inplace=True)
    df_clean['Married'].fillna(df_clean['Married'].mode()[0], inplace=True)
    df_clean['Dependents'].fillna(df_clean['Dependents'].mode()[0], inplace=True)
    df_clean['Self_Employed'].fillna(df_clean['Self_Employed'].mode()[0], inplace=True)
    df_clean['LoanAmount'].fillna(df_clean['LoanAmount'].median(), inplace=True)
    df_clean['Loan_Amount_Term'].fillna(df_clean['Loan_Amount_Term'].mode()[0], inplace=True)
    df_clean['Credit_History'].fillna(df_clean['Credit_History'].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        le = LabelEncoder()
        df_clean[column] = le.fit_transform(df_clean[column])
        label_encoders[column] = le

    # Split dataset into features and target
    X = df_clean.drop('Loan_Status', axis=1)
    y = df_clean['Loan_Status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train a model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, label_encoders, X.columns, accuracy

# Prediction function
def PredictionPage(model, scaler, label_encoders, feature_columns, accuracy):
    st.header("Loan Approval Prediction")
    st.subheader('Sir/Madam , YOU need to fill all necessary informations in order    to get a reply to your loan request !')

    # Function to handle encoding with unseen labels
    def encode_input(column, value, encoder):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            default_value = encoder.transform([encoder.classes_[0]])[0]
            return default_value

    tabs = st.tabs(["User Input", "Result"])

    with tabs[0]:
        # Get user input
        user_data = {}
        for column in feature_columns:
            if df[column].dtype == 'object' or df[column].nunique() <= 10:
                options = df[column].unique()
                user_data[column] = st.selectbox(f'Select {column}', options)
            else:
                user_data[column] = st.number_input(f'Enter {column}')

        # Allow user to upload image, video, or audio
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        uploaded_audio = st.file_uploader("Upload an audio", type=["mp3", "wav"])

        user_input_df = pd.DataFrame([user_data])

        # Preprocess user input
        for column in label_encoders.keys():
            if column in user_input_df.columns:
                user_input_df[column] = user_input_df[column].apply(lambda x: encode_input(column, x, label_encoders[column]))

        user_input_scaled = scaler.transform(user_input_df)

    with tabs[1]:
        # Make prediction with progress bar
        if st.button("Predict"):
            st.balloons()
            with st.spinner('Wait for it...'):    
                time.sleep(10)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            
            prediction = model.predict(user_input_scaled)
            result = "Approved" if prediction[0] == 1 else "Not Approved"
            st.write(f"The loan application is likely to be: {result}")
            
        # Display accuracy
        st.write(f"Model Accuracy: {accuracy}")

# Side bar function to handle navigation
def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Home", "Data Visualization", "Prediction"],
            icons=["house", "eye"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Home":
        HomePage()

    elif selected == "Data Visualization":
        ProgressBar()
        Graphs()

    elif selected == "Prediction":
        # Load model, scaler, and encoders
        model, scaler, label_encoders, feature_columns, accuracy = preprocess_and_train_model()
        PredictionPage(model, scaler, label_encoders, feature_columns, accuracy)

# Print side bar
sideBar()

# Footer
footer = """<style>
a:hover, a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
height:5%;
bottom: 0;
width: 100%;
background-color: #243946;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Shanika Mihirani <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Samir.s.s</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)



