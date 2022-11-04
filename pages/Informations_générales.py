import streamlit as st
import pandas as pd

# ------------------------ PAGE CONFIG ----------------------------------
st.set_page_config(
    page_title="About page",
    page_icon='🇧🇷',
    layout="wide"
)
st.title("About")

# ------------------------ DATA ----------------------------------
df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
# ------------------------ CONTENT ----------------------------------
with st.container():
    st.write("---")
    st.markdown("### Le dataset")
    st.markdown("* **Licence:** Libre\n"
                "* **Fournisseur:** Hospital Israelita Albert Einstein, at São Paulo, Brazil", unsafe_allow_html=True)
    st.download_button("Télécharger le dataset complet ici", data=df.to_csv().encode('utf-8'),
                       file_name="diagnosis_of_covid_2019.csv")

with st.container():
    st.write("---")
    st.markdown("### L'auteur")
    st.markdown("Je suis Frimpong ADOTRI, étudiant en Big Data et Machine Learning à EFREI-Paris. Je suis également apprenti "
                "Data Scientist à MUTEX. Je suis passionné de Data depuis l'âge de 16 ans et ce projet est un premier accomplissement"
                " dont je suis fier.")

with st.container():
    st.write("---")
    st.markdown("### Mon code Streamlit")
    st.markdown("Le code streamlit est un projet Pycharm disponible sous licence privée.")

















