import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List
import seaborn as sns

# ------------------------ PAGE CONFIG ----------------------------------
st.set_page_config(
    page_title="Analyse de forme",
    page_icon='🇧🇷',
    layout="wide"
)
st.title("Analyse de forme")

# ------------------------ DATA ----------------------------------
df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
boxplot_variables:List[str] = list(cleaned_df.select_dtypes(["float64"]).columns)
# ------------------------ CONTENT ----------------------------------
st.markdown("---")
st.markdown("### **Présentation**")
st.markdown("Il s'agit d'une exploration en surface du dataset à notre disposition. L'objectif est de prendre ses "
            "marques et se familiariser avec les différentes variables. Les principaux éléments qui s'en dégage sont :")
st.markdown("* **La variable à expliquer** : <span style=\"color:blue\">\"**_SARS-Cov-2 exam result_**\"</span>"
            "\n* **Les dimensions du dataset** : <span style=\"color:blue\">_5644 lignes x 111 colonnes_</span>"
            "\n* **Proportion globale de valeurs manquantes** : <span style=\"color:blue\">551682 (97.75 %)</span>"
            "\n * **Nature des variables** : <span style=\"color:blue\">70 variables **_«float64»_** | 37 variables **_«object»_** | 4 variables **_«int64»_**</span> ", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Focus sur les valeurs manquantes...")

with st.container():
    st.markdown("La **_heatmap_** suivante permet de visualiser la proportion de valeurs manquantes par colonne" 
            "dans le dataset. En effet, on remarque que les valeurs manquantes sont de couleur **blanche** tandis"
            " que les valeurs existantes sont en <span style=\"color:blue\">**bleu**</span>.\n", unsafe_allow_html=True)
    na_fig = px.imshow(df, width=1100, height=850, color_continuous_scale='bluered')
    na_fig.layout.coloraxis.showscale = False  #remove the imshow colorbar
    na_fig.update_layout(title_text='NaN values repartition per features', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
    st.write(na_fig)

with st.container():
    st.markdown("Étant donné  le grand nombre de variables (111) à notre disposition, il est primordial de nous"
                " débarasser des variables avec plus de 90% de valeurs manquantes. Après épuration de ces variables, on passe "
                "de 111 variables à 39 variables. La **heatmap** de notre nouveau dataset ressemble à ceci. Pour une meilleure"
                " compréhension, il faut savoir que les couleurs de la heatmap sont codées en octets. Le masque booléen sur "
                "lequel repose cette heatmap retourne **«1 si la valeur est une NaN»** et **«0 sinon»**. Les couleurs de la heatmap"
                " ne représentent que 2 valeurs opposées. Ainsi **0 = tous les"
                " à tous les chiffres de l'octet sont nuls** et **255=tous les chiffres de l'octet sont égaux à 1.** "
                "Donc toutes les couleurs égales à 0 représentent des valeurs existantes et les couleurs égales à 255, les NaN values.", unsafe_allow_html=True)
    na_fig = px.imshow(cleaned_df.isna(), width=900, height=900, )
    na_fig.layout.coloraxis.showscale = True  #remove the imshow colorbar
    na_fig.update_layout(title_text='Dataset after cleaning features with a high rate of NaN values', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
    st.write(na_fig)

with st.container():
    st.markdown("---")
    st.markdown("### **Statistiques descriptives**")
    st.markdown(r'''Les premières statistiques de nos variables sont révélatrices. On peut remarquer de prime abord
                que les variables sont toutes standardisées ($\mu = 0$  et  $\sigma = 1$) mise à part la variable
                **«Patient age quantile» (qui sont potentiellement des catégories d'âge)**, les variables 
                **«Mycoplasma pneumoniae»**, **«Partial thromboplastin time (PTT)»**, **«Urine-Sugar»**, 
                **«Prothrombin time (PT), Activity»** et **«D-Dimer» (qui ne possèdent que des valeurs manquantes)**
                 et la variable **«Fio2 (venous blood gas analysis)» (qui ne possède qu'une seule valeur qui est 
                 égale à 0)**.''')
    st.write("La variable **«Patient age quantile»** ")
    st.write("")
    st.markdown("<span style=\"color:red\">**Statistiques descriptives du dataset**</span>", unsafe_allow_html=True)
    st.dataframe(df.describe())
    st.write("")
    st.write(r'''Les **«boxplots»** nous permettent de visualiser la granularité de chacune de nos variables. 
    Globalement, les variables suivent une distribution relativement symétrique. En effet, la ligne médiane est
    centrée, le plus souvent dans la boîte à moustache. Par ailleurs, Les variables sont faiblement dispersées, ceci
     étant dû à la faible longueur des boîtes à moustache.''')
    st.markdown("Bien évidemment, quelques variables sont des exceptions. C'est le cas de la variable **«Eosinophils»** "
             "n'a pas du tout une distribution symétrique.")
    st.write("Pour finir, on peut remarquer que toutes les variables possèdent un nombre relativement important de "
             "valeurs extrêmes. Certaines variables comme **«Basophils»** ont des valeurs fortements extrêmes **(11.078 pour un 3e IQ de -0.387)**. ")
    box_fig = px.box(cleaned_df, y=boxplot_variables, notched=True)
    box_fig.update_layout(title_text='Numerical variables boxplot vizualisation', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                      width=850,
                      height=850,
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
    st.write(box_fig)

with st.container():
    st.markdown("---")
    st.markdown("### **Mon code Python**")
    code:str = '''
# DESCRIPTIVE ANALYSIS PYTHON CODE
    
# PYTHON MODULES
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

df:pd.DataFrame = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
# DISPLAY THE DATASET
print(df)

# NaN values rate and cleaning
print(f"NaN values rate = {df.isna().sum().sum()/df.shape[0]}")
na_fig = px.imshow(df, width=1000, height=750, title='NaN values repartition per features')
na_fig.show()
print((df.isna().sum()/df.shape[0]).sort_values(ascending=True))
cleaned_df:pd.DataFrame = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
print(cleaned_df)

# first statistics
print(df.describe())
    '''
    st.code(code, language='Python')



