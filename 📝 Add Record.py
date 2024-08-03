import streamlit as st
import pandas as pd
import seaborn as sns
from UI import *

st.set_page_config(page_title="Descriptive Analytics ", page_icon="📈", layout="wide")  
heading()

if 'number_of_rows' not in st.session_state:
    st.session_state['number_of_rows']=3
    st.session_state['type']='Categorical'
    

increment=st.sidebar.button('show more columns ➕')
if increment:
  st.session_state.number_of_rows +=1
decrement=st.sidebar.button('show fewer columns ➖')
if decrement:
 st.session_state.number_of_rows -=1

df=pd.read_csv('loan-train.csv')


 

theme_plotly = None # None or streamlit

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


st.markdown("##")

st.sidebar.header("Add New")
options_form=st.sidebar.form("Option Form")
Loan_ID = options_form.text_input("Loan_ID")
Gender=options_form.radio("Gender",{"Male","Female"})
Married=options_form.radio("Married",{"Yes","No"})
Education=options_form.radio("Education",{"Graduate","No Graduate"})
Self_Employed=options_form.radio("Self_Employed",{"Yes","No"})
ApplicantIncome=options_form.number_input("ApplicantIncome")
CoapplicantIncome=options_form.number_input("CoapplicantIncome")
LoanAmount=options_form.number_input("LoanAmount")
Loan_Amount_Term = options_form.number_input("Loan_Amount_Term", min_value=180, max_value=480, step=90)
Credit_History=options_form.selectbox("Credit_History",{"1","0"})
Property_Area=options_form.selectbox("Property_Area",{"Urban","Rural","Semiurban"})
Loan_Status=options_form.radio("Loan_Status",{"Y","N"})
add_data=options_form.form_submit_button(label="Add new record")

if add_data:
 if ApplicantIncome  !="" or  Property_Area!="":
     df = pd.concat([df, pd.DataFrame.from_records([{ 
         'Loan_ID': Loan_ID,
         'Gender':Gender,
         'Married':Married,
         'Education':Education,
         'Self_Employed':Self_Employed,
         'ApplicantIncome':int(ApplicantIncome),
         'CoapplicantIncome':int(CoapplicantIncome),
         'LoanAmount':int(LoanAmount),
         'Loan_Amount_Term':int(Loan_Amount_Term),
         'Credit_History':Credit_History,
         'Property_Area':Property_Area,
         'Loan_Status':Loan_Status,
         }])])
     try:
        df.to_csv("loan-train_csv",index=False)
     except:
        st.warning("Close dataset")
     st.success("New record has been added successfully !")
 else:
    st.sidebar.error("Loan Id is required")



#st.dataframe(df_selection,use_container_width=True)
shwdata = st.multiselect('Filter :', df.columns, default=["Loan_ID","Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"])
st.dataframe(df.tail(st.session_state['number_of_rows']),use_container_width=True,)

 
 