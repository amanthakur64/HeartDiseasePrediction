import streamlit as st
import streamlit_lottie
import requests
from streamlit_lottie import st_lottie
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time



st.set_page_config(page_title="Heart Disease Prediction", page_icon="heart",layout="centered")


with open('saved_model.pkl', 'rb') as file:
    data=pickle.load(file)

model=data['model']
le_smoking=data['le_smoking']
le_alcohol=data['le_alcohol']
le_stroke=data['le_stroke']
le_walk=data['le_walk']
le_sex=data['le_sex']
le_age=data['le_age']
le_diabetes=data['le_diabetes']
le_exercise=data['le_exercise']
le_health=data['le_health']
le_asthma=data['le_asthma']
le_kidney=data['le_kidney']
le_cancer=data['le_cancer']
scaler=data['scaler']



def load_animation(url):
    r=requests.get(url)
    return r.json()



lottie_code= load_animation("https://assets5.lottiefiles.com/private_files/lf30_8npirptd.json")

# HEADER
st.subheader("This is my Pattern Recognition & Machine Learning Course Bonus Project")
st.title("Heart Disease Prediction based on CDC's Behavioral Risk Factor Surveillance System")


with st.container():
    st.write("---")
    left_column,right_column = st.columns(2)
    with left_column:
        st.header("How it Works")
        st.write("Behind the web application, a machine learning model (Gradient boosting on Decision Trees) that has been trained on over 300,000 instances of survey data collected by the United States' CDC predicts the likelihood of a person having a heart disease.")


    with right_column:
        st_lottie(lottie_code,height=200,width=400,quality='high',key="tree")


st.write("---")

with st.container():

    left_column,right_column = st.columns(2)
    with left_column:
        st.header("Behavorial Survey")
        st.write("Fill the Short Survey below to get a prediction of your heart disease risk. Your data is not collected or stored anywhere.")
        st.write("---")


    with right_column:
        lottie_code= load_animation("https://assets4.lottiefiles.com/packages/lf20_tutvdkg0.json")
        st_lottie(lottie_code,height=200,width=400,quality='high',key="survey")

st.write("Are you Male or Female?")
checkbox_9=st.checkbox("Male",key="male")
checkbox_10=st.checkbox("Female",key="Female")
if checkbox_9:
    sex="Male"
elif checkbox_10:
    sex="Female"
else:
    sex="Female"

st.write("##")
age=st.select_slider("What is your Age Group?",options=["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"])
height=st.slider("What is your Height in cm?",1,250,1)
weight=st.slider("What is your Weight in kg?",1,250,1)
bmi=weight/((height/100)**2)

ok1 = st.button("Check your BMI")
if ok1:
    bmi=weight/((height/100)**2)
    st.write("Your BMI is:",bmi)


st.write("##")
st.write("Have you smoked over 100 cigarettes in your life?")
checkbox_1 = st.checkbox("Yes",key="checkbox_1")
checkbox_2 = st.checkbox("No",key="checkbox_2")
if checkbox_1:
    cig = "Yes"
elif checkbox_2:
    cig = "No"
else:
    cig = "No"



st.write("##")
st.write("Do you Consume Alcoholic Drinks Heavily? (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week")
checkbox_3 = st.checkbox("Yes",key="checkbox_3")
checkbox_4 = st.checkbox("No",key="checkbox_4")
if checkbox_3:
    alc="Yes"
elif checkbox_4:
    alc="No"
else:
    alc="No"


st.write("##")
st.write("Have you ever had a Stroke?")
checkbox_5=st.checkbox("Yes",key="checkbox_5")
checkbox_6=st.checkbox("No",key="checkbox_6")
if checkbox_5:
    stroke="Yes"
elif checkbox_6:
    stroke="No"
else:
    stroke="No"



st.write("##")
st.write("For how many days during the past 30 days was your physical health not good? (Injury/Illness)")
phy_health=st.slider("Days",0,30,1,key="phy_health")



st.write("##")
st.write("For how many days during the past 30 days was your mental health not good?")
mental_health=st.slider("Days",0,30,1,key="mental_health")



st.write("##")
st.write("Do you have difficulty walking or climbing stairs?")
checkbox_7=st.checkbox("Yes",key="checkbox_7")
checkbox_8=st.checkbox("No",key="checkbox_8")
if checkbox_7:
    walk="Yes"
elif checkbox_8:
    walk="No"
else:
    walk="No"


st.write("##")
st.write("Have you ever been diagnosed with any of the following?")
checkbox_11=st.checkbox("Diabetes",key="diabetes")
checkbox_18=st.checkbox("Yes (during pregnancy)",key="preg")
checkbox_17=st.checkbox("No, borderline diabetes",key="bd")
checkbox_12=st.checkbox("Asthma",key="asthma")
checkbox_13=st.checkbox("Kidney Disease",key="kidney")
checkbox_14=st.checkbox("Skin Cancer",key="cancer")

if checkbox_17:
    diabetes="No, borderline diabetes"


if checkbox_18:
    diabetes="Yes (during pregnancy)"


if checkbox_11:
    diabetes="Yes"
else:
    diabetes="No"


if checkbox_12:
    asthma="Yes"
else:
    asthma="No"


if checkbox_13:
    kidney="Yes"
else:
    kidney="No"


if checkbox_14:
    cancer="Yes"
else:
    cancer="No"


st.write("##")
st.write("Did you perform any sort of significant physical activity/exercise other than regular work in the past 30 days?")
checkbox_15=st.checkbox("Yes",key="checkbox_30")
checkbox_16=st.checkbox("No",key="checkbox_31")
if checkbox_15:
    excercise="Yes"
elif checkbox_16:
    excercise="No"
else:
    excercise="No"



st.write("##")
st.write("How many hours of sleep do you usually get in a 24hr time period?")
sleep=st.slider("Hours",0,24,1,key="sleep")



st.write("##")
st.write("How would you rate your overall health?")
health=st.select_slider("Choose an option",options=["Poor","Fair","Good","Very Good","Excellent"],key="health")


ok2 = st.button("Check Your Heart's Health")
if ok2:
    df=[bmi,cig,alc,stroke,phy_health,mental_health,walk,sex,age,diabetes,excercise,health,sleep,asthma,kidney,cancer]
    columns=['BMI',	'Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','DiffWalking','Sex','AgeCategory','Diabetic','PhysicalActivity','GenHealth','SleepTime','Asthma','KidneyDisease','SkinCancer']
    df= pd.DataFrame([df],columns=columns)

    df['Smoking']=le_smoking.transform(df['Smoking'])
    df['AlcoholDrinking']=le_alcohol.transform(df['AlcoholDrinking'])
    df['Stroke']=le_stroke.transform(df['Stroke'])
    df['DiffWalking']=le_walk.transform(df['DiffWalking'])
    df['Sex']=le_sex.transform(df['Sex'])
    df['AgeCategory']=le_age.transform(df['AgeCategory'])
    df['Diabetic']=le_diabetes.transform(df['Diabetic'])
    df['PhysicalActivity']=le_exercise.transform(df['PhysicalActivity'])
    df['GenHealth']=le_health.transform(df['GenHealth'])
    df['Asthma']=le_asthma.transform(df['Asthma'])
    df['KidneyDisease']=le_kidney.transform(df['KidneyDisease'])
    df['SkinCancer']=le_cancer.transform(df['SkinCancer'])
  

    normalisecolumns=['BMI','PhysicalHealth','MentalHealth','SleepTime']
    df[normalisecolumns] = scaler.transform(df[normalisecolumns])

    pred=model.predict(df)
    percentpred=model.predict_proba(df)
    with st.container():
        st.write("---")
        left_column,mid,right_column = st.columns(3)

        with right_column:
            lottie_code= load_animation("https://assets5.lottiefiles.com/packages/lf20_GZVTNZ.json")
            st_lottie(lottie_code,height=200,width=200,quality='high',key="heart_load")

        with left_column:
            st.header("Calculating...")
            time.sleep(1)
            st.write("Feeding Data to Model...")
            time.sleep(1)
            st.write("Making Predictions...")
            time.sleep(1)
            st.write("Retrieving Data...")
            time.sleep(1)

    
    if pred[0]==0:
        with st.container():
            st.write("---")
            left_column,mid,right_column = st.columns(3)

            with right_column:
                lottie_code= load_animation("https://assets2.lottiefiles.com/packages/lf20_afs4kbqm.json")
                st_lottie(lottie_code,height=200,width=200,quality='high',key='good')

            with left_column:
                st.subheader("Your Heart's Health is Good!")
                st.write("Good Job! The probability of you developing a heart disease is: ",percentpred[0][1]*100,"%")

                


            

        
    else: 
        with st.container():
            st.write("---")
            left_column,mid,right_column = st.columns(3)

            with right_column:
                lottie_code= load_animation("https://assets3.lottiefiles.com/private_files/lf30_yABbl9.json")
                st_lottie(lottie_code,height=200,width=200,quality='high',key="bad")

            with left_column:
                st.subheader("Uh oh! Your Heart's Health is not Good!")
                st.write("Do not panic, Seek Medical Help. The probability of you developing a heart disease is: ",percentpred[0][1]*100,"%")

    time.sleep(2)
    st.write("---")
    github_icon=("https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png")
    github_link=("https://github.com/amanthakur64")
    st.markdown(f"""<a href={github_link}><img src={github_icon} alt="drawing" width="50"/></a>""",unsafe_allow_html=True,)





