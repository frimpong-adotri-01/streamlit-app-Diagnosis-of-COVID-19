import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.figure_factory as ff

# ------------------------ DATA ----------------------------------
df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
nb_positifs:pd.DataFrame = df[df["SARS-Cov-2 exam result"]=="positive"].dropna(axis=1).shape[0]
nb_negatifs:pd.DataFrame = df[df["SARS-Cov-2 exam result"]=="negative"].dropna(axis=1).shape[0]
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
cleaned_df.drop("Patient ID", axis=1, inplace=True)
categories = pd.DataFrame((((df.isna().sum()/df.shape[0]))).sort_values(ascending=True), columns=["NaN_rate"])
viral_rate:pd.DataFrame = categories[(categories["NaN_rate"]>.7) & (categories["NaN_rate"]<.86)]
blood_tests:pd.DataFrame = categories[(categories["NaN_rate"]>.87) & (categories["NaN_rate"]<.9)]
positive:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "positive"]
negative:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "negative"]
viral_rate_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.75) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.88)]
blood_tests_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.87) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.9)]


# ------------------------ PAGE CONFIG ----------------------------------

st.set_page_config(
    page_title="Diagnosis of COVID-19 app",
    page_icon='🇧🇷',
    layout="wide"
)
with st.spinner("Un instant s'il vous plaît !"):
    time.sleep(1)

st.title("Diagnosis of COVID-19 and its clinical spectrum 🇧🇷")

st.markdown("---")

# ------------------------ KPI METRICS ----------------------------------
st.markdown("### <span style=\"color:blue\">Principaux KPI</span> \n", unsafe_allow_html=True)
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total des individus", df.shape[0])
kpi2.metric("Nombre de cas négatifs", nb_negatifs, delta=f"{(nb_negatifs/df.shape[0])*100:.2f} %")
kpi3.metric("Nombre de cas positifs", nb_positifs, delta=f"{(nb_positifs/df.shape[0])*100:.2f} %", delta_color='inverse')

# ------------------------ THE DATAFRAME ----------------------------------
st.markdown("---")
st.markdown("### <span style=\"color:blue\">Le Dataset</span>", unsafe_allow_html=True)
st.dataframe(df)

# ------------------------ SOME PLOTS ----------------------------------
st.markdown("---")
st.markdown("### <span style=\"color:blue\">Visualisation des variables</span> \n", unsafe_allow_html=True)

blood_selector = st.selectbox("Tests sanguins", blood_tests_columns)
blood_fig = ff.create_distplot([cleaned_df[blood_selector].dropna().values], [blood_selector], colors= ['red'], bin_size=.2,
    show_rug=False).update_layout(title=f'Distplot de {blood_selector}', title_x=.5, title_font_color='red', xaxis_title=f"{blood_selector}")
st.write(blood_fig)



viral_selector = st.selectbox("Taux viraux", cleaned_df.select_dtypes("object").columns)
viral_fig = px.pie(cleaned_df, cleaned_df[viral_selector].dropna(),
                   color_discrete_sequence=px.colors.qualitative.G10,
                   title=f'Pie Chart de {viral_selector}').update_layout(title_x=.5, title_font_color='red')
st.write(viral_fig)

st.markdown("---")
st.markdown("### <span style=\"color:blue\">Relations entre tests sanguins</span> \n", unsafe_allow_html=True)


blood_radio = st.radio("Type de graphe", ('Scatter plot', 'Scatter plot habillé'))
blood_x = st.selectbox("Tests sanguins : Abscisse", blood_tests_columns)
blood_y = st.selectbox("Tests sanguins : Ordonnée", blood_tests_columns)

if blood_radio == 'Scatter plot':
    fig = px.scatter(x=cleaned_df[blood_x], y=cleaned_df[blood_y],color_discrete_sequence=px.colors.qualitative.Set1).update_layout(
        title=f'Scatter plot {blood_x}/{blood_y}', title_x=.5, title_font_color='red', xaxis_title=f"{blood_x}", yaxis_title=f"{blood_y}")
    st.write(fig)
if blood_radio == 'Scatter plot habillé':
    habillage = st.selectbox("Taux viraux : Abscisse", cleaned_df.select_dtypes("object").columns)
    fig = px.scatter(x=cleaned_df[blood_x], y=cleaned_df[blood_y], color=cleaned_df[habillage]).update_layout(
        title=f'Scatter plot {blood_x}/{blood_y}', title_x=.5, title_font_color='red', xaxis_title=f"{blood_x}", yaxis_title=f"{blood_y}")
    st.write(fig)

st.markdown("---")
st.markdown("### <span style=\"color:blue\">Relations entre taux viraux</span> \n", unsafe_allow_html=True)


viral_x = st.selectbox("Taux viraux : Abscisse", cleaned_df.select_dtypes("object").columns)
viral_y = st.selectbox("Taux viraux : Ordonnée", cleaned_df.select_dtypes("object").columns)
fig = px.imshow(pd.crosstab(cleaned_df[viral_x], cleaned_df[viral_y]), text_auto=True).update_layout(
        title=f'Crosstab {viral_x}/{viral_y}', title_x=.5, title_font_color='red', xaxis_title=f"{viral_x}", yaxis_title=f"{viral_y}")
fig.layout.coloraxis.showscale = False
st.write(fig)