import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# ------------------------ PAGE CONFIG ----------------------------------

st.set_page_config(
    page_title="Analyse de fond",
    page_icon='🇧🇷',
    layout="wide"
)
st.title("Analyse de fond")


# ------------------------ DATA ----------------------------------
df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
cleaned_df.drop("Patient ID", axis=1, inplace=True)
categories = pd.DataFrame((((df.isna().sum()/df.shape[0]))).sort_values(ascending=True), columns=["NaN_rate"])
viral_rate:pd.DataFrame = categories[(categories["NaN_rate"]>.7) & (categories["NaN_rate"]<.86)]
blood_tests:pd.DataFrame = categories[(categories["NaN_rate"]>.87) & (categories["NaN_rate"]<.9)]
positive:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "positive"]
negative:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "negative"]
viral_rate_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.75) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.88)]
blood_tests_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.87) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.9)]

def test_statistique(variable:str, alpha:float) -> str:
  statistic, p_value = ttest_ind(negative.sample(positive.shape[0], random_state=0)[variable].dropna(), positive[variable].dropna())
  if p_value < alpha:
    return "reject H0"
  else:
    return "fail to reject H0"

def test_statistique_dependance(variable:str, alpha:float) -> str:
  cross_table = pd.crosstab(cleaned_df["SARS-Cov-2 exam result"], cleaned_df[variable])
  result = chi2_contingency(cross_table)
  p_value = result[1]
  if p_value < alpha:
    return "reject H0"
  else:
    return "fail to reject H0"

# ------------------------ CONTENT ----------------------------------
with st.container():
    st.markdown("---")
    st.markdown("### **Présentation**")
    st.markdown("Il s'agit ici d'une analyse plus approfondie du dataset. De l'analyse des relations entre les"
                " variables à l'explication des variables en passant par les tests d'hypothèses, nous passerons ce dataset au peigne fin."
                " Dans la mesure où nous serions amenés à développer un modèle de **machine learning**, il est important de noter que la"
                " variable cible (Target variable) serait <span style=\"color:green\">\"**_SARS-Cov-2 exam result_**\"</span>. Par ailleurs, "
                "notons que nous sommes en présence de **classes deséquilibrées** dans la target variable. En effet, on dispose "
                "d'une importante proportion d'individus étiquettés <span style=\"color:blue\">\"**negative**\"</span> (90%) contre une "
                "assez faible proportion d'individus étiquettés <span style=\"color:red\">\"**posistive**\"</span> (10%).", unsafe_allow_html=True)
    sars_cov_fig = px.pie(df, "SARS-Cov-2 exam result", width=500, height=500, color_discrete_sequence=px.colors.qualitative.G10,
                          title='Rate of Positive/Negative COVID cases')
    sars_cov_fig.update_layout(title_text='Répartition des classes de la Target variable', #center figure title
                        title_x=.5,
                        font_family="Courier New",
                        font_color="black",
                        title_font_family="Arial",
                        title_font_color="red",  #title color
                        legend_title_font_color="green"   #legend color
                        )
    st.write(sars_cov_fig)
    st.markdown("Dans ce cas de **classes deséquilibrées**, les métriques les plus optimales pour développer le modèle peuvent être le"
            " **score F1** ou encore le **recall**.")

with st.container():
    st.markdown("---")
    st.markdown("### **Informations cachées dans les variables**")
    st.write("")
    st.markdown("##### **Les variables numériques «float64»**")
    st.markdown("Elles possèdent globalement une moyenne avoisinant zéro et "
                "un écart-type avoisinant un (1) (Voir **Analyse de forme**). Par conséquent, ces variables sont déjà standardisées. "
                "Notons également que ces variables suivent presque toutes une distribution normale.")
    with st.expander("Voir les graphiques de distribution des variables"):
        for column in cleaned_df.select_dtypes("float64"):
            dist_fig = ff.create_distplot([cleaned_df[column].dropna().values], [column], bin_size=.2, show_rug=False)
            dist_fig.update_layout(title_text=f'{column} distribution', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                     xaxis_title=f"{column}",
                     yaxis_title="density",
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
            st.write(dist_fig)
    st.write("")

    st.markdown("##### **Les variables catégorielles «object»**")
    st.markdown("Ces variables n'étant pas numériques, il est préférable d'observer les différentes classes "
                "composant ces variables:")
    for column in cleaned_df.select_dtypes("object"):
        st.markdown(f"<span style=\"color:blue\">**{column:-<70}**  **{cleaned_df[column].dropna().unique()}**</span>", unsafe_allow_html=True)
    st.markdown("On recense un important nombre de classe binaires par variable, en faisant abstraction des **NaN values**. "
                "En général, les classes **«negative»** et **«not_detected»** sont majoritairement écrasantes en défaveur la classe opposée. "
                "Ainsi, par analogie à la **variable Target** qui est elle-même catégorielle, ces variables possèdent des classes "
                "deséquilibrées. Cependant, la variable **«Rhinovirus/Enterovirus»** est assez intéressante dans la mesure où "
                "on peut considérer que ce n'est relativement pas une classe deséquilibrée.")
    with st.expander("Voir les camemberts de répartition des classes des variables"):
        for column in cleaned_df.select_dtypes("object"):
            classes_fig = px.pie(df, df[column].dropna(), color_discrete_sequence=px.colors.qualitative.G10)
            classes_fig.update_layout(title_text=f'{column}  positive/negative cases',  # center figure title
                              title_x=.5,
                              font_family="Courier New",
                              font_color="black",
                              title_font_family="Arial",
                              title_font_color="red",  # title color
                              legend_title_font_color="green"  # legend color
                              )
            st.write(classes_fig)
    st.write("")

    st.markdown("##### **Les variables de type «int64»**")
    st.markdown("On distingue :")
    st.markdown("* **\"Patient age quantile\"**: c'est une variable particulère dans la mesure où on ne "
            "possède aucune information sur elle. On retrouve beaucoup d'hypothèses spéculatives sur cette "
            "variable dans le forum **kaggle** dédié. Cette variable peut alors être interprétée comme une "
            "**«variable catégorielle encodée ordinalement»** (d'où ses valeurs numériques) et ses valeurs peuvent être associées"
            " à des tranches d'âge de 5 (0 -> [1 à 5 ans], 1 -> [6 à 10 ans], ...). Mais encore une fois, **il ne s'agit que d'hypothèses**.")
    quantile_fig = px.bar(cleaned_df, x=cleaned_df["Patient age quantile"].dropna().value_counts().index, y=cleaned_df["Patient age quantile"].dropna().value_counts().values)
    quantile_fig.update_layout(title_text=f'Patient age quantile repartition',  # center figure title
                               title_x=.5,
                               font_family="Courier New",
                               xaxis_title=f"Patient age quantile",
                               yaxis_title="count",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
    st.write(quantile_fig)
    st.markdown("* **\"Patient addmited to regular ward (1=yes, 0=no)\" | \"Patient addmited to semi-intensive unit (1=yes, 0=no)\"** "
            "**| \"Patient addmited to intensive care unit (1=yes, 0=no)\"**: le libellé de ces variables nous permet"
            " de supposer qu'elles sont également des **«variables encodées ordinalement»**. De plus ces variables possèdent aussi"
                " des classes fortement deséquilibrées.")
    with st.expander("Voir les répartitions des variables encodées"):
        for column in ("Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)"):
            fig = px.pie(cleaned_df, df[column].dropna())
            fig.update_layout(title_text=f'{column}  positive/negative cases',  # center figure title
                              title_x=.5,
                              font_family="Courier New",
                              font_color="black",
                              title_font_family="Arial",
                              title_font_color="red",  # title color
                              legend_title_font_color="green"  # legend color
                              )
            st.write(fig)

with st.container():
    st.markdown("---")
    st.markdown("### **Relations inter-variables et data visualization**")
    st.markdown(
        "Dans un premier temps, nous allons découper nos variables en 2 grands groupes. Sur base de résultats de recherches"
        " internet croisées avec les informations récoltées sur **kaggle**, ce découpage se base par coïncidence sur la "
        "proportion de **NaN values**. En effet, on dégage 2 grandes catégories : **Les tests viraux** (0.75 < NaN < 0.86),"
        " **les taux sanguins** (0.87 < NaN < 0.9). Les variables de type **«float64»** sont des variables associées aux "
        "**taux sanguins** et les variables de type **«object»** aux **tests viraux**. La data vizualisation nous permettra "
        "d'observer visuellement nos variables afin de tirer nos conclusions.")
    st.write("")
    cat1, cat2 = st.columns(2)
    with cat1:
        st.markdown("<span style=\"color:red\">**Tests viraux**</span>", unsafe_allow_html=True)
        st.dataframe(viral_rate)
    with cat2:
        st.markdown("<span style=\"color:red\">**Taux sanguins**</span>", unsafe_allow_html=True)
        st.dataframe(blood_tests)
    st.write("")
    st.markdown("* **Relations Target/Taux sanguins:** Pour observer cette relation, nous allons superposer entre elles, les **distribution"
                " plots** des **taux sanguins** pour chaque classe de la **target variable**. La plus part des **taux sanguins** suivent sensiblement la "
                "même distribution indépendamment des cas postifs ou négatifs. Cependant, les variables **«Platelets»**, **«Leukocytes»** et **«Monocytes»** "
                "sont intéressantes dans la mesure où la distribution présente un comportement spécifique selon les cas positifs ou négatifs. On peut "
                "alors émettre une hypothèse à tester plus tard. \n<span style=\"color:blue\">**<u>Hypothèse N°1</u>:** \"Les variables **«Platelets»**, **«Leukocytes»** et **«Monocytes»** ont une une incidence sur le résultat d'un individu au test COVID\".</span>", unsafe_allow_html=True)
    with st.expander("Voir les graphiques de relation Target/Taux sanguins"):
        for column in blood_tests_columns:
            rel1_fig = ff.create_distplot([positive[column].dropna().values, negative[column].dropna().values], ["positive case", "negative case"], bin_size=.2, show_rug=False)
            rel1_fig.update_layout(title_text=f'{column}  positive/negative cases', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                     xaxis_title=f"{column}",
                     yaxis_title="density",
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
            st.write(rel1_fig)
    st.write("")
    st.markdown("* **Relations Target/Tests viraux:** Ces deux types de variables étant des variables catégorielles, on ne peut comparer "
                "leurs distributions. Pour comparer des variables catégorielles entre elles, nous aurons recours à une **«crosstab»** qui résulte"
                " du croisement des variables deux à deux. Nous visualiserons les résultats de la crosstab avec des heatmaps. Les conclusions tirées "
                "de l'observation des heatmaps montrent qu'en général, les personnes positives aux tests de taux viraux sont négatives au COVID et inversement. "
                "Ainsi, globalement, un individu n'est pas infecté par plus d'un virus. Par ailleurs, on remarque que la variable **«Rhinovirus/Enterovirus»** "
                "est assez prépondérante parmis les individus négatifs au COVID-19. Peut-on en déduire que les individus atteints du **«Rhinovirus/Enterovirus»**"
                " sont moins prédisposés à contracter le COVID-19 ? C'est une hypothèse à tester."
                "\n<span style=\"color:blue\">**<u>Hypothèse N°2</u>:** \"La variable **«Rhinovirus/Enterovirus»** préserve-t-il de la contraction du COVID ?\"</span>", unsafe_allow_html=True)
    with st.expander("Voir les graphiques de relation Target/Tests viraux"):
        for column in viral_rate_columns:
            rel2_fig = px.imshow(pd.crosstab(cleaned_df["SARS-Cov-2 exam result"], cleaned_df[column]), text_auto=True)
            rel2_fig.layout.coloraxis.showscale = False  #remove the imshow colorbar
            rel2_fig.update_layout(title_text=f'{column}  positive/negative cases', #center figure title
                     title_x=.5,
                     font_family="Courier New",
                     xaxis_title=f"{column}",
                     yaxis_title="SARS-Cov-2 exam result",
                     font_color="black",
                     title_font_family="Arial",
                     title_font_color="red",  #title color
                     legend_title_font_color="green"   #legend color
                     )
            st.write(rel2_fig)
    st.markdown("* **Relations Target/Patient age quantile:** Toujours en supposant que la variable **«Patient age quantile»** correspond "
                "aux tranches d'âges, on visualise via un **countplot** les répartitions d'individus négatifs et positifs par tranche d'âge."
                " Sous l'hypothèse de catégortisation des âges, on remarque que plus on est âgé, plus on a de cas postifs dans la catégorie."
                "Cependant, la proportion de cas positifs fluctue à partir de la catégorie 5 (i.e. entre 21 et 25 ans).")
    quantile_fig = px.histogram(cleaned_df, x="Patient age quantile", color="SARS-Cov-2 exam result", barmode="group", width=850, color_discrete_sequence=px.colors.qualitative.G10)
    quantile_fig.update_layout(title_text=f'Patient age quantile countplot',  # center figure title
                               title_x=.5,
                               font_family="Courier New",
                               xaxis_title=f"Patient age quantile",
                               yaxis_title="count",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
    st.write(quantile_fig)

with st.container():
    st.markdown("---")
    st.markdown("### Analyse détaillée")
    st.markdown("Après avoir réalisé une analyse de fond précédemment, il convient d'approfondir les résultats trouvés et tester et approuver "
                "ou rejeter certaines hypothèses.")
    st.write("")
    st.markdown("##### **Relations entre les variables de tests sanguins**")
    st.markdown("Pour ce faire, visualisons la matrice de corrélation de ces variables. On note que les variables les plus corrélées sont : ")
    blood_corr_fig = px.imshow(cleaned_df[blood_tests_columns].corr(), width=850, height=850)
    blood_corr_fig.update_layout(title_text=f'Correlation entre les variables de tests sanguins',  # center figure title
                               title_x=.5,
                               font_family="Courier New",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
    st.write(blood_corr_fig)
    st.write("")
    st.markdown(
        "En réalisant une matrice de **scatter plots**, on remarque que les graphes de ces différentes variables, les unes par rapport "
        "aux autres, suivent toujours une tendance linéaire. De plus un graphe 3D en prenant en compte les 3 variables sur les axes confirme "
        "cette tendance linéaire.")
    st.markdown("**Nota:** Les variables **Hemoglobin** et **Hematocrit** sont très fortement correlées (97% environ de correlation).")
    with st.expander("Voir les tendances inter-variables."):
        blood_corr_scatter_fig = px.scatter_matrix(cleaned_df[['Hemoglobin', 'Hematocrit', 'Red blood Cells']], width=850, height=850)
        blood_corr_scatter_fig.update_layout(title_text=f'Corralated variables tendance',  # center figure title
                                            title_x=.5,
                                            font_family="Courier New",
                                            font_color="black",
                                            title_font_family="Arial",
                                            title_font_color="red",  # title color
                                            legend_title_font_color="green"  # legend color
                                            )
        st.write(blood_corr_scatter_fig)
    scatter3d_blood_corr = px.scatter_3d(cleaned_df, 'Hemoglobin', 'Hematocrit', 'Red blood Cells', color="SARS-Cov-2 exam result", width=850, height=850, color_discrete_sequence=px.colors.qualitative.G10)
    scatter3d_blood_corr.update_layout(title_text=f'Corralated variables tendance (3D view)',  # center figure title
                                            title_x=.5,
                                            font_family="Courier New",
                                            font_color="black",
                                            title_font_family="Arial",
                                            title_font_color="red",  # title color
                                            legend_title_font_color="green"  # legend color
                                            )
    scatter3d_blood_corr.update_traces(marker=dict(size=7,
                                        line=dict(width=1)),
                                        selector=dict(mode='markers'))

    st.write(scatter3d_blood_corr)

    st.markdown("##### **Relations entre les variables de taux viraux**")
    st.markdown(
        "Pour ce faire,  nous allons nous attarder sur les variables **Influenza A, rapid test** et **Influenza B, rapid test** "
        "qui, curieusement pourrait s'apparenter  aux variables **Influenza A** et **Influenza B**. Réalisons une «crosstab» à cet"
        " effet : ")
    influenza_a = px.imshow(pd.crosstab(cleaned_df["Influenza A, rapid test"], cleaned_df["Influenza A"]), text_auto=True)
    influenza_a.layout.coloraxis.showscale = False  # remove the imshow colorbar
    influenza_a.update_layout(title_text=f'{column}  Crosstab Influenza A, rapid test/Influenza A',  # center figure title
                           title_x=.5,
                           font_family="Courier New",
                           #xaxis_title="Influenza A, rapid test",
                           #yaxis_title="Influenza A",
                           font_color="black",
                           title_font_family="Arial",
                           title_font_color="red",  # title color
                           legend_title_font_color="green"  # legend color
                           )
    st.write(influenza_a)

    influenza_b = px.imshow(pd.crosstab(cleaned_df["Influenza B, rapid test"], cleaned_df["Influenza B"]),
                            text_auto=True)
    influenza_b.layout.coloraxis.showscale = False  # remove the imshow colorbar
    influenza_b.update_layout(title_text=f'{column}  Crosstab Influenza B, rapid test/Influenza B',
                              # center figure title
                              title_x=.5,
                              font_family="Courier New",
                              # xaxis_title="Influenza B, rapid test",
                              # yaxis_title="Influenza B",
                              font_color="black",
                              title_font_family="Arial",
                              title_font_color="red",  # title color
                              legend_title_font_color="green"  # legend color
                              )
    st.write(influenza_b)
    st.markdown("Les tests **Influenza A, rapid test** détectent 15 tests jugés négatifs par les tests **Influenza A**. De même, les"
                " tests **Influenza B, rapid test** sont négatifs à 18 tests jugés positifs par les tests **Influenza B**. Ces résultats"
                " s'expliquent par la mauvaise sensibilité des **rapids tests** et à la fiabilité des tests dits **classiques**. "
                "Cet article donne plus d'informations à ce sujet : "
                "[article](https://www.cdc.gov/flu/professionals/diagnosis/overview-testing-methods.htm#:~:text=Most%20of%20the%20rapid%20influenza,improved%20accuracy%2C%20including%20higher%20sensitivity.)")


with st.container():
    st.markdown("---")
    st.markdown("### Tests d'hypothèses")
    st.markdown("<span style=\"color:blue\">**<u>Hypothèse N°1</u>:** \"Les variables **«Platelets»**, **«Leukocytes»** et **«Monocytes»** ont une une incidence sur le résultat d'un individu au test COVID\".</span>", unsafe_allow_html=True)
    st.markdown("On effectue ici un test de student pour confirmer/infirmer l'hypothèse.")
    st.markdown(r'''$H_{0}$ : Les quantités de moyenne de **«Platelets»**, **«Leukocytes»** et **«Monocytes»** sont les mêmes pour les patients négatifs comme postitifs.''')
    st.markdown(r'''$H_{1}$ : Les quantités de moyenne de **«Platelets»**, **«Leukocytes»** et **«Monocytes»** fluctuent en fonction des  patients négatifs et postitifs.''')
    st.markdown("On va effectuer un test statistique d'indépendance sur la moyenne par p-value et pour ce faire, nous allons rééquilibrer nos classes deséquilibrées. Nous allons créer un échantillon d'individus négatifs "
                "dont la taille est égale à la taille d'échantillon des individus positifs. Ces individus négatifs seront sélectionnés aléatoirement. Cependant, ce caractère aléatoire modifiera"
                "à chaque exécution du code, les résultats des tests. Pour uniformiser les résultats d'une exécution à l'autre, la graine du générateur aléatoire sera fixée(argument **random_state=0**). "
                "Le niveau de significativité fixé sera de **1%**. De plus, la conclusion sera tirée en fonction de la valeur de la p-value. Si la p-value est inférieure au niveau de significativité,"
                ", on rejette l'hypothèse nulle, sinon, on échoue à rejeter l'hypothèse nulle. Le code python ci-dessous illustre le procédé:")
    st.code("""
from scipy.stats import ttest_ind
import pandas as pd

df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
cleaned_df.drop("Patient ID", axis=1, inplace=True)
positive:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "positive"]
negative:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "negative"]
blood_tests_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.88) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.9)]

def test_statistique(variable:str, alpha:float) -> str:
    statistic, p_value = ttest_ind(negative.sample(positive.shape[0], random_state=0)[variable].dropna(), positive[variable].dropna())
    if p_value < alpha:
        return "reject H0"
    else:
        return "fail to reject H0"

for column in blood_tests_columns:
    print(f"{column:-<70} {test_statistique(column, .01)}")
    """, language="Python")
    st.markdown(r'''Les résultats du test statistique prouvent bien qu'on rejette $ H_{0} $ au profit de $ H_{1} $ pour les variables qui nous intéressent. On vient
    ainsi de prouver, au niveau de significativité $ \alpha=0.01 $, que les quantités de moyenne de **«Platelets»**, **«Leukocytes»** 
    et **«Monocytes»** fluctuent en fonction des  patients négatifs et postitifs.''')
    st.markdown("**Nota**: La variable **Eosinophils** à l'issue du test possède les même propriétés que les variables **Leukocytes**,... Le graphe de la distribution de sa relation avec la target peut appuyer ce résultat du test statistique.")
    for column in blood_tests_columns:
        if not(column in ("Monocytes", "Leukocytes", "Platelets")):
            st.markdown(f"<span style=\"color:green\">**{column:-<70} {test_statistique(column, .01)}**</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style=\"color:red\">**{column:-<70} {test_statistique(column, .01)}**</span>",unsafe_allow_html=True)
    st.write("")
    st.markdown("<span style=\"color:blue\">**<u>Hypothèse N°2</u>:** \"La variable **«Rhinovirus/Enterovirus»** préserve-t-il de la contraction du COVID ?\".</span>",
        unsafe_allow_html=True)
    st.markdown("On effectue ici un test de student pour confirmer/infirmer l'hypothèse.")
    st.markdown(r'''$H_{0}$ : La variable **«Rhinovirus/Enterovirus»** préserve de la contraction du COVID.''')
    st.markdown(
        r'''$H_{1}$ : La variable **«Rhinovirus/Enterovirus»** ne préserve pas de la contraction du COVID.''')
    st.markdown(
        "On va effectuer un test statistique de dépendance **\"khi-deux\" (pour les variables catégorielles)** par p-value. Le seul moyen de réalisé ce test a été de conserver les deséquilibres entre les classes. "
        "Le niveau de significativité fixé sera de **1%**. Nous sommes en présence de variables catégorielles, donc pour effectuer ce test, nous allons avoir recours "
        "à la **crosstab** utilisée plus haut. Cette **crosstab** sera le tableau sur lequel le test de **\"khi-deux\"** s'appuiera pour calculer la **statistique de test**, "
        "la **p-value** et le **degré de liberté**. On s'intéressera à la p-value. Ainsi, la conclusion sera tirée en fonction de la valeur de la p-value. Si la p-value est inférieure au niveau de significativité,"
        ", on rejette l'hypothèse nulle, sinon, on échoue à rejeter l'hypothèse nulle. Le code python ci-dessous illustre le procédé:")
    st.code("""
from scipy.stats import chi2_contingency
import pandas as pd

positive:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "positive"]
negative:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "negative"]
viral_rate_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.75) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.88)]

def test_statistique_dependance(variable:str, alpha:float) -> str:
    cross_table = pd.crosstab(cleaned_df["SARS-Cov-2 exam result"], cleaned_df[variable])
    result = chi2_contingency(cross_table)
    p_value = result[1]
    if p_value < alpha:
        return "reject H0"
    else:
        return "fail to reject H0"

for column in viral_rate_columns:
    print(f"{column:-<70} {test_statistique_dependance(column, .01)}")
    """, language='Python')
    st.markdown(r'''Les résultats du test statistique prouvent bien qu'on rejette $ H_{0} $ au profit de $ H_{1} $ pour les variables qui nous intéressent. On vient
        ainsi de prouver, au niveau de significativité $ \alpha=0.01 $, que la variable **«Rhinovirus/Enterovirus»** ne préserve pas de la contraction du COVID.''')
    for column in viral_rate_columns:
        if not(column == "Rhinovirus/Enterovirus"):
            st.markdown(f"<span style=\"color:green\">**{column:-<70} {test_statistique_dependance(column, .01)}**</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style=\"color:red\">**{column:-<70} {test_statistique_dependance(column, .01)}**</span>",unsafe_allow_html=True)

with st.container():
    st.markdown('---')
    st.markdown("### Mon code Python")
    code:str = """
# IMPORTS
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

# NETTOYAGE DES DONNÉES
df = pd.read_excel("https://github.com/frimpong-adotri-01/datasets/blob/main/dataset.xlsx?raw=true")
cleaned_df = df[df.columns[(df.isna().sum()/df.shape[0]) < .9]]
cleaned_df.drop("Patient ID", axis=1, inplace=True)

# CATÉGORISATION DES VARIABLES
categories = pd.DataFrame((((df.isna().sum()/df.shape[0]))).sort_values(ascending=True), columns=["NaN_rate"])
viral_rate:pd.DataFrame = categories[(categories["NaN_rate"]>.7) & (categories["NaN_rate"]<.77)]
blood_tests:pd.DataFrame = categories[(categories["NaN_rate"]>.8) & (categories["NaN_rate"]<.9)]
positive:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "positive"]
negative:pd.DataFrame = cleaned_df[cleaned_df["SARS-Cov-2 exam result"] == "negative"]
viral_rate_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.75) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.88)]
blood_tests_columns = cleaned_df.columns[(cleaned_df.isna().sum()/cleaned_df.shape[0]>.88) & (cleaned_df.isna().sum()/cleaned_df.shape[0]<.9)]

# REPRÉSENTATION DES CLASSES DE LA TARGET VARIABLE
sars_cov_fig = px.pie(df, "SARS-Cov-2 exam result", width=500, height=500, color_discrete_sequence=px.colors.qualitative.G10, title='Rate of Positive/Negative COVID cases')
sars_cov_fig.update_layout(title_text='NaN values repartition per features', 
                          title_x=0.5,  #center figure title
                          font_family="Courier New",
                          font_color="black",
                          title_font_family="Arial",
                          title_font_color="red",  #title color
                          legend_title_font_color="green"   #legend color
                          )
sars_cov_fig.show()

# DISTRIBUTION DES VARIABLES «float64» (DISTPLOT)
for column in cleaned_df.select_dtypes("float64"):
    dist_fig = ff.create_distplot([cleaned_df[column].dropna().values], [column], bin_size=.2, show_rug=False)
    dist_fig.update_layout(title_text=f'{column} distribution', 
                            title_x=0.5,  #center figure title
                            font_family="Courier New",
                            xaxis_title=f"{column}",
                            yaxis_title="density",
                            font_color="black",
                            title_font_family="Arial",
                            title_font_color="red",  #title color
                            legend_title_font_color="green"   #legend color
              )
    dist_fig.show()

# VARIABLES CATÉGORIELLES DE TYPE «object»
    # CLASSES
for column in cleaned_df.select_dtypes("object"):
    print(f"{column:-<70} {cleaned_df[column].dropna().unique()}")
    # PIE PLOT POUR VISUALISER LA RÉPARTITION DES CLASSES DANS LES VARIABLES
for column in cleaned_df.select_dtypes("object"):
    classes_fig = px.pie(df, df[column].dropna(), color_discrete_sequence=px.colors.qualitative.G10)
    classes_fig.update_layout(title_text=f'"{column}" classes repartition',  # center figure title
                              title_x=0.5,
                              font_family="Courier New",
                              font_color="black",
                              title_font_family="Arial",
                              title_font_color="red",  # title color
                              legend_title_font_color="green"  # legend color
                              )
    classes_fig.show()

# VARIABLE "Patient age quantile" (BAR PLOT DE REPRÉSENTATION)
quantile_fig = px.bar(cleaned_df, x=cleaned_df["Patient age quantile"].dropna().value_counts().index, y=cleaned_df["Patient age quantile"].dropna().value_counts().values)
quantile_fig.update_layout(title_text=f'Patient age quantile distribution',  # center figure title
                               title_x=0.5,
                               font_family="Courier New",
                               xaxis_title=f"Patient age quantile",
                               yaxis_title="count",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
quantile_fig.show()

# RÉPARTITION DES CLASSES DANS LES AUTRES VARIABLES DE TYPE «int64»
for column in ("Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)"):
    fig = px.pie(cleaned_df, df[column].dropna())
    fig.update_layout(title_text=f'{column} classes repartition',  # center figure title
                      title_x=0.5,
                      font_family="Courier New",
                      font_color="black",
                      title_font_family="Arial",
                      title_font_color="red",  # title color
                      legend_title_font_color="green"  # legend color
                      )
    fig.show()

# DISTRIBUTION PLOT REPRÉSENTATIVE DE LA RELATION TARGET/TAUX SANGUINS
for column in blood_tests_columns:
    rel1_fig = ff.create_distplot([positive[column].dropna().values, negative[column].dropna().values], ["positive case", "negative case"], bin_size=.2, show_rug=False)
    rel1_fig.update_layout(title_text=f'{column}  positive/negative cases', #center figure title
                          title_x=0.5,
                          font_family="Courier New",
                          xaxis_title=f"{column}",
                          yaxis_title="density",
                          font_color="black",
                          title_font_family="Arial",
                          title_font_color="red",  #title color
                          legend_title_font_color="green"   #legend color
                          )
    rel1_fig.show()

# DISTRIBUTION PLOT REPRÉSENTATIVE DE LA RELATION TARGET/TESTS VIRAUX
for column in viral_rate_columns:
    rel2_fig = px.imshow(pd.crosstab(cleaned_df["SARS-Cov-2 exam result"], cleaned_df[column]), text_auto=True)
    rel2_fig.layout.coloraxis.showscale = False  #remove the imshow colorbar
    rel2_fig.update_layout(title_text=f'{column}  positive/negative cases', #center figure title
                          title_x=0.5,
                          font_family="Courier New",
                          xaxis_title=f"{column}",
                          yaxis_title="SARS-Cov-2 exam result",
                          font_color="black",
                          title_font_family="Arial",
                          title_font_color="red",  #title color
                          legend_title_font_color="green"   #legend color
                          )
    rel2_fig.show()

# COUNTPLOT DE LA VARIABLE "Patient age quantile"
quantile_fig = px.histogram(cleaned_df, x="Patient age quantile", color="SARS-Cov-2 exam result", barmode="group", width=850, color_discrete_sequence=px.colors.qualitative.G10)
quantile_fig.update_layout(title_text=f'Patient age quantile countplot',  # center figure title
                               title_x=0.5,
                               font_family="Courier New",
                               xaxis_title=f"Patient age quantile",
                               yaxis_title="count",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
quantile_fig.show()

# RELATIONS ENTRE LES VARIABLES DE TESTS SANGUINS
    # MATRICE DE CORRELATION
blood_corr_fig = px.imshow(cleaned_df[blood_tests_columns].corr(), width=850, height=850)
blood_corr_fig.update_layout(title_text=f'Correlation between blood tests variables',  # center figure title
                               title_x=0.5,
                               font_family="Courier New",
                               font_color="black",
                               title_font_family="Arial",
                               title_font_color="red",  # title color
                               legend_title_font_color="green"  # legend color
                               )
blood_corr_fig.show()
    # SCATTER MATRIX POUR LES VARIABLES FORTEMENT CORRELÉES
blood_corr_scatter_fig = px.scatter_matrix(cleaned_df[['Hemoglobin', 'Hematocrit', 'Red blood Cells']], width=850, height=850)
blood_corr_scatter_fig.update_layout(title_text=f'Corralated variables tendance',  # center figure title
                                            title_x=0.5,
                                            font_family="Courier New",
                                            font_color="black",
                                            title_font_family="Arial",
                                            title_font_color="red",  # title color
                                            legend_title_font_color="green"  # legend color
                                            )
blood_corr_scatter_fig.show()
    # SCATTER PLOT 3D POUR VISUALISER LES 3 VARIABLES SIMULTANÉMENT
scatter3d_blood_corr = px.scatter_3d(cleaned_df, 'Hemoglobin', 'Hematocrit', 'Red blood Cells', color="SARS-Cov-2 exam result", width=850, height=850, color_discrete_sequence=px.colors.qualitative.G10)
scatter3d_blood_corr.update_layout(title_text=f'Corralated variables tendance (3D view)',  # center figure title
                                            title_x=0.5,
                                            font_family="Courier New",
                                            font_color="black",
                                            title_font_family="Arial",
                                            title_font_color="red",  # title color
                                            legend_title_font_color="green"  # legend color
                                            )
scatter3d_blood_corr.update_traces(marker=dict(size=7,
                                    line=dict(width=1)),
                                    selector=dict(mode='markers'))
scatter3d_blood_corr.show()

# RELATIONS ENTRE LES VARIABLES DE TAUX VIRAUX


# TEST STATISTIQUE DE STUDENT
def test_statistique(variable:str, alpha:float) -> str:
  statistic, p_value = ttest_ind(negative.sample(positive.shape[0], random_state=0)[variable].dropna(), positive[variable].dropna())
  if p_value < alpha:
    return "reject H0"
  else:
    return "fail to reject H0"
    
for column in blood_tests_columns:
    print(f"{column:-<70} {test_statistique(column, .01)}")

# TEST STATISTIQUE DE KHI-DEUX
def test_statistique_dependance(variable:str, alpha:float) -> str:
  cross_table = pd.crosstab(cleaned_df["SARS-Cov-2 exam result"], cleaned_df[variable])
  result = chi2_contingency(cross_table)
  p_value = result[1]
  if p_value < alpha:
    return "reject H0"
  else:
    return "fail to reject H0"

print("")
for column in viral_rate_columns:
        print(f"{column:-<70} {test_statistique_dependance(column, .01)}")
    """
    st.code(code, language='Python')



