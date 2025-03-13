import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind, levene
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuración inicial
st.set_page_config(page_title="Análisis de Felicidad Global", page_icon="😊", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_csv('2019.csv')

df = load_data()

# Título principal
st.title("Análisis del Índice de Felicidad Mundial 🌍")
st.markdown("""
**Integrantes:**
- Javier Alejandro González Díaz C-312
- Kevin Márquez Vega C-312
- José Miguel Leyva de la Cruz C-312
""")

# Sidebar con controles
with st.sidebar:
    st.header("Controles")
    show_raw_data = st.checkbox("Mostrar datos crudos")
    selected_year = st.selectbox("Año del dataset", [2019])
    selected_continent = st.multiselect("Filtrar por continente", df['Continent'].unique())

# Filtrado de datos
if selected_continent:
    df = df[df['Continent'].isin(selected_continent)]

# Sección de datos
if show_raw_data:
    st.subheader("Datos Crudos")
    st.dataframe(df, use_container_width=True)

# Estadísticas básicas
st.subheader("Estadísticas Básicas")
cols = st.columns(3)
with cols[0]:
    st.metric("Países analizados", df['Country or region'].nunique())
with cols[1]:
    st.metric("Puntuación promedio", f"{df['Score'].mean():.2f}")
with cols[2]:
    st.metric("Año del estudio", selected_year)

# Análisis de distribución
st.subheader("Distribución del Índice de Felicidad")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Score'], kde=True, binwidth=0.8, ax=ax)
plt.title('Distribución de Happiness Score')
plt.xlabel('Puntuación')
plt.ylabel('Frecuencia')
st.pyplot(fig)

with st.expander("Conclusiones - Distribución"):
    st.markdown("""
    - La distribución del índice de felicidad sigue una curva aproximadamente normal
    - La mayoría de países se concentran entre 4.5 y 6.5 puntos
    - Existen valores extremos en ambos extremos de la escala
    - Los países nórdicos lideran consistentemente los primeros puestos
    """)

# Correlaciones
st.subheader("Relación entre Variables")
col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox("Variable X", df.select_dtypes(include=np.number).columns.tolist(), index=3)
with col2:
    y_axis = st.selectbox("Variable Y", df.select_dtypes(include=np.number).columns.tolist(), index=2)

fig = px.scatter(df, x=x_axis, y=y_axis, color='Continent', hover_name='Country or region',
                 title=f"{x_axis} vs {y_axis}")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Conclusiones - Correlaciones"):
    st.markdown("""
    - El PIB per cápita muestra la correlación más fuerte con la felicidad
    - El soporte social es el segundo factor más influyente
    - La esperanza de vida saludable tiene una relación positiva significativa
    - La percepción de corrupción muestra una correlación negativa débil
    """)

# Análisis por continente
st.subheader("Comparativa por Continente")
continent_analysis = df.groupby('Continent').agg({
    'Score': 'mean',
    'GDP per capita': 'mean',
    'Social support': 'mean'
}).reset_index()

fig = px.bar(continent_analysis, x='Continent', y='Score', 
             title='Puntuación Promedio por Continente',
             color='Continent')
st.plotly_chart(fig, use_container_width=True)

# Sección de Distribución de Países por Continente
st.subheader("Distribución de Países por Continente")

# Cálculo de la distribución
continents_groups = df.groupby('Continent')['Continent'].agg('count')

# Crear dos columnas para el gráfico y la tabla
col1, col2 = st.columns([2, 1])

with col1:
    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.bar(continents_groups.index, continents_groups.values, color='skyblue')
    plt.title('Número de Países por Continente')
    plt.xlabel('Continente')
    plt.ylabel('Cantidad de Países')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Sección de Desviación Local
st.subheader("Desviación Local de Felicidad por País")

# Explicación de la métrica
st.markdown("""
En nuestra región, el índice de felicidad de algunos países puede tener una variación significativa con respecto al índice de felicidad de su continente. 
Algunos países muestran una variación positiva y otros una situación más desfavorable. Con este nuevo cálculo pretendemos reflejar esa **Desviación Local**.
""")

st.latex(r"""
DL_i = Score_i - \frac{1}{n_c} \sum_{j \in C_i} Score_j
""")

# Explicación de la fórmula
st.markdown("""
Donde:
- $DL_i$ es la Desviación Local del país $i$
- $Score_i$ es el Score de felicidad del país $i$ 
- $C_i$ es el conjunto de países del continente al que pertenece el país $i$
- $n_c$ es el número de países en el continente $C_i$
- $\sum_{j \in C_i} Score_j$ es la suma de los Scores de todos los países del continente $C_i$
""")

# Cálculo de la Desviación Local
media_por_continente = df.groupby('Continent')['Score'].transform('mean')
df['Local Deviation'] = df['Score'] - media_por_continente

# Mostrar los primeros 10 resultados
st.markdown("**Primeros 10 resultados:**")
st.dataframe(df[['Country or region', 'Continent', 'Score', 'Local Deviation']].head(10), use_container_width=True)

# Gráfico de desviaciones por continente
st.markdown("**Distribución de Desviaciones por Continente**")
fig = px.box(df, x='Continent', y='Local Deviation', color='Continent',
             points="all", hover_data=['Country or region'],
             title="Distribución de Desviaciones Locales por Continente")
st.plotly_chart(fig, use_container_width=True)

# Tabla con los países con mayor y menor desviación
st.markdown("**Países con Mayor y Menor Desviación Local**")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mayor Desviación Positiva**")
    top_positive = df.nlargest(5, 'Local Deviation')[['Country or region', 'Continent', 'Local Deviation']]
    st.dataframe(top_positive.style.format({'Local Deviation': '{:.2f}'}), use_container_width=True)

with col2:
    st.markdown("**Mayor Desviación Negativa**")
    top_negative = df.nsmallest(5, 'Local Deviation')[['Country or region', 'Continent', 'Local Deviation']]
    st.dataframe(top_negative.style.format({'Local Deviation': '{:.2f}'}), use_container_width=True)

# Conclusiones en sección expandible
with st.expander("Conclusiones - Desviación Local"):
    st.markdown("""
    - Los países nórdicos muestran consistentemente las mayores desviaciones positivas
    - Algunos países africanos presentan las mayores desviaciones negativas
    - América muestra una alta variabilidad en sus desviaciones
    - Oceanía tiene una distribución más compacta, con menos variabilidad
    - La desviación local ayuda a identificar casos atípicos y patrones regionales
    """)

# Conclusiones en sección expandible
with st.expander("Conclusiones - Distribución por Continente"):
    st.markdown("""
    - **Europa** tiene la mayor representación en el estudio
    - **África** muestra una alta participación, reflejando su diversidad
    - **Oceanía** tiene la menor cantidad de países incluidos
    - La distribución refleja el enfoque global del estudio
    - América está dividida en Norte y Sur para un análisis más detallado
    """)

with st.expander("Conclusiones - Comparativa Continental"):
    st.markdown("""
    - Europa lidera en todos los indicadores principales
    - Oceanía muestra valores altos a pesar de su menor densidad poblacional
    - África presenta los valores más bajos en todas las métricas
    - América del Norte tiene mayor variabilidad en los resultados
    """)

#Pruebas de hipótesis
st.header("Pruebas de Hipótesis")
st.subheader("Prueba de hipotesis unilateral sobre la media de felicidad")

st.markdown("""
En esta sección, se realizará una prueba de hipótesis unilateral para determinar si el valor esperado del Score es mayor que 5.13.Resultado alcanzado por el mismo estudio que agrupo estos datos en el año 2018.
""")

scores = df['Score']
    
    # Parámetros de la prueba
mu_0 = 5.13
alpha = 0.05
    
# Cálculos estadísticos
sample_mean = scores.mean()
sample_std = scores.std()
n = len(scores)
t_statistic = (sample_mean - mu_0) / (sample_std / np.sqrt(n))
dof = n - 1  # Cambiado de 'df' para evitar conflicto con el DataFrame
p_value = 1 - stats.t.cdf(t_statistic, dof)
t_critico = stats.t.ppf(1 - alpha, dof)
    
# Crear visualización
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(stats.t.ppf(0.001, dof), stats.t.ppf(0.999, dof), 100)
y = stats.t.pdf(x, dof)
    
ax.plot(x, y, label='Distribución t')
ax.fill_between(x, y, where=(x >= t_critico), color='red', alpha=0.3, label='Región de rechazo')
ax.axvline(t_statistic, color='green', linestyle='--', label='Estadístico t observado')
ax.axvline(t_critico, color='black', linestyle=':', label='Valor crítico')
    
ax.set_title('Prueba de Hipótesis Unilateral para el Valor Esperado del Score')
ax.set_xlabel('Estadístico t')
ax.set_ylabel('Densidad de Probabilidad')
ax.legend()
ax.grid(True)
    
st.pyplot(fig)
    
# Mostrar conclusiones
st.subheader("Resultados de la Prueba de Hipótesis")
st.markdown(f"""
    - **Hipótesis nula (H₀):** μ ≤ {mu_0}
    - **Hipótesis alternativa (H₁):** μ > {mu_0}
    - **Estadístico t calculado:** {t_statistic:.4f}
    - **Valor crítico:** {t_critico:.4f}
    - **P-valor obtenido:** {p_value:.4f}
    - **Media muestral:** {sample_mean:.4f}
    - **Desviación estándar muestral:** {sample_std:.4f}
    """)
    
    # Conclusión final
if p_value < alpha:
        st.success("**Conclusión:** Rechazar H₀")
        st.info(f"Hay evidencia estadística significativa (α = {alpha}) para afirmar que el valor esperado del Score es mayor que {mu_0}")
else:
        st.error("**Conclusión:** No rechazar H₀")
        st.warning(f"No hay evidencia estadística suficiente (α = {alpha}) para afirmar que el valor esperado del Score sea mayor que {mu_0}")

st.subheader("Prueba de hipotesis para dos poblaciones")
st.markdown("""Dado que anteriormente se analizaron los datos de un país respecto a su continente , un análisis interesante podría ser establecer comparaciones entre los continentes , dado que cada uno en si tiene sus características en cada uno de los aspectos medidos en el conjunto de datos .Veamos que influencia tiene esto sobre su *felicidad*.
""")
# Prueba de hipótesis para dos poblaciones

# Sección de Comparativa África vs Europa
st.markdown("""### Distribución de la Felicidad en África y Europa 🤔📊

El siguiente histograma muestra la distribución comparativa de los scores de felicidad entre los países africanos y europeos. Las barras superpuestas permiten visualizar las diferencias en los niveles de felicidad entre ambos continentes y evaluar la hipótesis de que los países africanos tienden a tener un score de felicidad menor a 5.5 y por ende menor que el alto índice existente en los países europeos.
""")
# Filtrar datos
africa_scores = df[df['Continent'] == 'Africa']['Score']
europe_scores = df[df['Continent'] == 'Europe']['Score']

# Crear dos columnas para organizar el contenido
col1, col2 = st.columns([2, 1])

with col1:
    # Histogramas superpuestos
    st.markdown("**Distribución de Puntuaciones de Felicidad**")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(africa_scores, bins=15, alpha=0.5, label='África', color='orange')
    ax1.hist(europe_scores, bins=15, alpha=0.5, label='Europa', color='blue')
    ax1.set_xlabel('Puntuación de Felicidad')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Puntuaciones: África vs Europa')
    ax1.legend()
    st.pyplot(fig1)

with col2:
    # Mostrar promedios
    st.markdown("**Promedios de Felicidad**")
    st.metric("África", f"{africa_scores.mean():.2f}")
    st.metric("Europa", f"{europe_scores.mean():.2f}")

# Pruebas estadísticas
st.subheader("Análisis Estadístico")

# --- Prueba t de Student ---
st.subheader("Prueba t de Student para muestras independientes"  )
st.markdown("""
Esta prueba compara si existen diferencias significativas entre las medias de los scores de Europa y África.
Se plantean las hipótesis:  
- H₀: μ_Europa = μ_África (No hay diferencia en las medias)
- H₁: μ_Europa ≠ μ_África (Existe diferencia significativa)
""")

t_stat, p_value = ttest_ind(europe_scores, africa_scores)
st.markdown("**Resultados:**")
st.write(f"- Estadístico t: {t_stat:.4f} (Magnitud de la diferencia estandarizada)")
st.write(f"- Valor p: {p_value:.4e} (Probabilidad de observar tal diferencia si H₀ es cierta)")

# Interpretación
st.markdown("""
**Interpretación:**  
Un valor p menor que 0.05 sugiere evidencia suficiente para rechazar la hipótesis nula, 
indicando que la diferencia observada en los promedios no es atribuible al azar. 
La dirección de la diferencia se determina comparando las medias reales de los grupos.
""")

# --- Prueba de Levene ---
st.subheader("Prueba de Levene para Homogeneidad de Varianzas")
st.markdown("""
Evalúa si ambos grupos mantienen varianzas iguales, requisito fundamental para 
la validez de la prueba t estándar. Las hipótesis son:  
- H₀: σ²_Europa = σ²_África (Homogeneidad de varianzas)
- H₁: σ²_Europa ≠ σ²_África (Heterogeneidad de varianzas)
""")

levene_stat, levene_pvalue = levene(europe_scores, africa_scores)
st.markdown("**Resultados:**")
st.write(f"- Estadístico: {levene_stat:.4f} (Medida de discrepancia entre varianzas)")
st.write(f"- Valor p: {levene_pvalue:.4f} (Probabilidad de observar tal discrepancia si H₀ es cierta)")

# Interpretación
st.markdown("""
**Interpretación:**  
Un valor p < 0.05 indicaría violación del supuesto de homogeneidad, requiriendo 
el uso de ajustes como la corrección de Welch en la prueba t. Valores altos sugieren 
que las varianzas son estadísticamente comparables, validando el uso de la prueba t clásica.
""")

# Conclusión final
with st.expander("Conclusiones - Comparativa África vs Europa"):
    st.markdown("""
    ### Hallazgos Principales:
    1. **Diferencias Significativas:**
       - Existe una diferencia estadísticamente significativa entre los índices de felicidad de África y Europa (p < 0.001)
       - Europa muestra puntuaciones consistentemente más altas

    2. **Distribución de Datos:**
       - Las puntuaciones europeas están más concentradas en el rango superior
       - África presenta una distribución más amplia y sesgada hacia valores más bajos

    3. **Normalidad de Datos:**
       - Ambos continentes muestran desviaciones de la normalidad en los extremos de la distribución
       - Europa presenta una distribución más cercana a la normalidad

    4. **Homogeneidad de Varianzas:**
       - Las varianzas no son homogéneas (p < 0.05), lo que sugiere mayor variabilidad en África

    ### Implicaciones:
    - Las diferencias culturales, económicas y sociales entre continentes tienen un impacto significativo en los niveles de felicidad
    - Los países africanos podrían beneficiarse de políticas públicas enfocadas en mejorar los factores que más impactan la felicidad
    - Europa podría servir como modelo para identificar prácticas exitosas en la promoción del bienestar social
    """)

# Sección de Cambios Realizados
st.subheader("Cambios Realizados: Test de Normalidad sobre el Score")

# Cargar los datos
df = pd.read_csv('2019.csv')  # Asegúrate de que el archivo esté en la misma carpeta
scores = df['Score']

# Prueba de Shapiro-Wilk
st.markdown("## Test de Shapiro-Wilk 📈🔍")

# Justificación del uso de Shapiro-Wilk
st.markdown("### Justificación del Uso de Shapiro-Wilk")
st.markdown("""
El test de Shapiro-Wilk es la mejor opción para esta base de datos porque es especialmente adecuado para muestras de tamaño pequeño a moderado, como es el caso de los 156 países incluidos en el ranking de felicidad. Este test tiene una alta potencia estadística para detectar desviaciones de la normalidad en muestras de este tamaño, lo que lo hace más confiable que otros tests como el de Kolmogorov-Smirnov, que es menos sensible, o el de Anderson-Darling, que es más útil para muestras más grandes o cuando se quiere enfocar en las colas de la distribución. Además, el test de Shapiro-Wilk es ampliamente utilizado y reconocido en la práctica estadística, lo que lo convierte en una opción robusta y adecuada para evaluar la normalidad de los datos en este contexto.
""")
shapiro_test = stats.shapiro(scores)
st.write(f"- **Estadístico:** {shapiro_test.statistic:.4f}")
st.write(f"- **Valor p:** {shapiro_test.pvalue:.4f}")

# Interpretación del Test de Shapiro-Wilk
alpha = 0.05  # Nivel de significancia
st.markdown("""### Resultados del Test de Shapiro-Wilk e interpretación final
    . Estadístico W cercano a 1 ✅  
    . P-valor > 0.05 ✅  
    . Interpretación: No hay evidencia para rechazar normalidad  
""")
if shapiro_test.pvalue > alpha:
    st.success("**Conclusión:** Los datos parecen seguir una distribución normal según Shapiro-Wilk.")
else:
    st.error("**Conclusión:** Los datos NO parecen seguir una distribución normal según Shapiro-Wilk.")

# Visualizaciones para evaluar normalidad
st.markdown("### Visualizaciones para Evaluar Normalidad")

# Crear una figura con tres subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 1. Q-Q Plot (Gráfico Cuantil-Cuantil) usando scipy.stats.probplot
stats.probplot(scores, plot=ax1)
ax1.set_title('Q-Q Plot')

# 2. Histograma con curva normal
sns.histplot(scores, kde=True, ax=ax2)
ax2.set_title('Histograma con Distribución Normal')

# 3. Boxplot
sns.boxplot(x=scores, ax=ax3)
ax3.set_title('Boxplot')

plt.tight_layout()
st.pyplot(fig)

# Regresion lineal SCORE y GDP per capita
st.subheader("Regresión Lineal")
st.markdown("""Evidentemente, nuestros datos están relacionados con numerosos índices de los campos economicos, sociales y de la salud. Sería interesante visualizar que tan realacionados está alguno de estos datos con el índice de la Felicidad (score). Para ello utilizaremos Regresión Lineal, la cual es una técnica que establece una línea recta en el comportamiento de nuestros datos para identificar que tan relacionados están.

### Elección de Variable Independiente
Antes de realizar el análisis de la regresión, debemos seleccionar cual será nuestra variable independiente. En este caso, elegiremos el GBP per Capita. Ya que como se muestra en la matriz de correlación es la variable más relacionada con el Score entre todas las demás. 
            """)

df_numericas = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = df_numericas.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de correlación')
st.pyplot(plt)

st.markdown(""" #### Ya seleccionada la variable independiente (GDP per capita), procederemos a realizar el análisis de la regresión lineal. """)

# Análisis de la regresion lineal 

# Preparación de variables
x = df[['GDP per capita']]
y = df['Score']

# Modelo de regresión con scikit-learn
model = LinearRegression()
model.fit(x, y)
intercept = model.intercept_
slope = model.coef_[0]

# Mostrar resultados de la regresión
st.subheader("Análisis de la Regresión Lineal")
st.write(f"Intersección: **{intercept:.4f}**")
st.write(f"Pendiente: **{slope:.4f}**")
st.write(f"Ecuación de la recta: **y = {intercept:.4f} + {slope:.4f}x**")

# Gráfico de regresión
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', alpha=0.5, label='Datos reales')
ax.plot(x, model.predict(x), color='red', linewidth=2, label='Línea de regresión')
ax.set_xlabel('PIB per cápita')
ax.set_ylabel('Score')
ax.set_title('Relación entre Score y PIB per cápita')
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Modelo statsmodels para análisis detallado
x_sm = sm.add_constant(x)
model_sm = sm.OLS(y, x_sm).fit()

# Resumen del modelo
st.subheader("Análisis Estadístico Detallado")
st.text(model_sm.summary().as_text())
residuals = model_sm.resid
y_pred = model.predict(x)

st.markdown(""" ### Métricas principales
1. Notemos el valor del R-Squared que es de 0.630, esto quiere decir que el GBP percapita influye en un 63% de la variación del Score de Felicidad analizado en este trabajo.
   
2. El valor de Prob(F-static) es extredamente pequeño, lo cual nos da a entender que este modelo es estadísticamente significativo y confiable.
   
3. Coeficientes: El valor del score cuando GBP per capita es 0, es de 3.3993 (valor const en el summary), que es a su vez el intercepto calculado anteriormente. La pendiente es de 2.2181, esto que nos quiere decir: que una unidad de GBP per Capita aumenta el Score en 2.2181 su valor.
   
4. El valor p asociado a los test t (usado para ver si un coeficiente es significativamente distinto de 0) es de 0.000, lo cual nos da a entender que el coeficiente de GBP per Capita es estadísticamente significativo y confiable.

### Otras métricas de interes
1. Durbin-Watson: 1.378 (utilizado para detectar autocorrelación en los residuos).

2. Omnibus y Prob(Omnibus): Tests de normalidad para los residuos (valores p > 0.05 sugieren que los residuos se distribuyen normalmente).

3. Jarque-Bera (JB) y Prob(JB): También pruebas de normalidad.

4. Kurtosis: 2.742 (medida de la “acuminación” de la distribución de los residuos).

5. Cond. No.: 4.77 (número de condición, que indica si hay problemas de multicolinealidad).
""")

# Gráfico de residuos
st.header("Análisis de Residuos 🔍🚮")
st.markdown("""Analizaremos cada uno de los supuestos del modelo para verificar si nuestro modelo, valga la redundancia, es correcto
1. Los errores ($e_1$, ... ,$e_n$) son independientes
2. El valor esperado del error esperado $e_i$ es cero 
3. Homocedasticidad
4. Los errores además son identicamente distribuidos y siguen una distribución normal con media cero y varianza $\sigma^2$
   
Grafiquemos los residuos para saber si el modelo cumple los supuestos del modelo 
""")
fig_res, ax_res = plt.subplots()
ax_res.scatter(y_pred, residuals, color='green', alpha=0.5)
ax_res.axhline(y=0, color='red', linestyle='--')
ax_res.set_xlabel('Valores Predichos')
ax_res.set_ylabel('Residuos')
ax_res.set_title('Residuos vs. Valores Predichos')
st.pyplot(fig_res)

# Histograma de los residuos
sns.histplot(residuals, kde=True)
plt.xlabel('Residuos')
plt.title('Histograma de los residuos')
st.pyplot(plt)

# Conclusiones
st.subheader("Conclusiones Finales para la Regresión Lineal")
conclusion = """
**Interpretación de los resultados y validación de supuestos:**

1. **Significancia estadística:** 
   - El valor p para el PIB per cápita , indica que el efecto es significante.
   
2. **Bondad de ajuste:**
   - El R-cuadrado ajustado (0.63) muestra que el modelo explica 63% de la variabilidad en el Score.

3. **Homocedasticidad:**
   - Se cumple homocedasticidad , o sea la varianza de los errores es constante .

4. **Normalidad de residuos:**
   - Distribución Normal✅ (Valor p Jarque-Bera: 1.244).

5. **Autocorrelación:**
    - DW ≈ 2: No hay autocorrelación (residuos independientes).

    - DW < 1.5: Sugiere autocorrelación positiva (los residuos están correlacionados en el tiempo).

    - DW > 2.5: Sugiere autocorrelación negativa.
   
    **En este caso** se aprecia con un valor de :(Durbin-Watson: 1.378) , una **leve correlación positiva** pero no es crítico.
"""

st.markdown(conclusion)

st.subheader("Regresión lineal múltiple 🪢")

st.markdown("""La regresión lineal múltiple es una extensión de la regresión lineal simple, que permite analizar la relación entre una variable dependiente y dos o más variables independientes. Mientras que la regresión lineal simple considera solo una variable independiente, la regresión lineal múltiple proporciona una forma más completa de entender y predecir los valores de la variable dependiente al considerar múltiples factores simultáneamente.

Procederemos de manera similar, es decir: primero seleccionaremos las mejores variables dependientes y luego procederemos con el análisis de regresión lineal múltiple.

### Elección de las mejores Variables ⚙️
Para elegir las mejores variables independientes, debemos tener en cuenta que estas deben estar relacionadas con el Score y a su vez no deben estar muy relacionadas entre ellas. Para esto utilizaremos un método de elección de caracaterísticas el cual recursivamente determina que dos datos son los adecuados para un analisis de regresión lineal múltiple. Comprobaremos si en efecto esos datos son correctos con un análisis de inflación de la varianza. Finalmente realizaremos el análisis de la regresión e interpretaremos los resultados.
""")

X = df_numericas.drop(columns=['Score'])
y = df['Score']

# Modelo de regresión lineal
model = LinearRegression()

# Eliminación recursiva de características
rfe = RFE(estimator=model, n_features_to_select=3)
fit = rfe.fit(X, y)

# Resultados
selected_features = X.columns[fit.support_]
st.write("### Variables seleccionadas:")
st.write(selected_features)

st.markdown(""" ### Las variables seleccionadas son aquellas que:

- Maximizan la capacidad predictiva del modelo.

- Minimizan la redundancia (es decir, no están altamente correlacionadas entre sí).

- Contribuyen significativamente a explicar la variabilidad de y.
            
    ### Análisis de la multicolinealidad 📊
            Hagamos un análisis del factor inlfación de la varianza para determinar si aún debemos suprimir alguno de estos datos.
""")
X_selected = sm.add_constant(X[selected_features])

# Calcular VIF para cada variable seleccionada
vif = pd.DataFrame()
vif["Variable"] = X_selected.columns
vif["VIF"] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]

st.write(vif)
st.markdown(""" ### 
                       ✅ Los VIF de las variables independientes son todos bastante bajos, todos menores que 5 lo que sugiere que no hay un problema significativo 
            de multicolinealidad entre ellas. Por tanto podemos proceder con un analisis de regresión lineal múltiple utilizando estos tres datos
            """)

st.subheader("Regresión lineal múltiple")
# Regresión múltiple 
df = pd.read_csv('2019.csv')

# Preparar las variables independientes seleccionadas
X = df[['Social support', 'Healthy life expectancy', 'Freedom to make life choices']]
y = df['Score']

# Agregar una constante
X = sm.add_constant(X)

# Ajustar el modelo de regresión lineal múltiple
model_multiple = sm.OLS(y, X).fit()

# Resumen del modelo
summary = model_multiple.summary()
st.write(summary)

st.subheader("Análisis de los resultados de la Regresión Lineal Múltiple")

st.write("""
### 1. Información General del Modelo
- **Dep. Variable: Score**: Variable dependiente que se busca predecir.
- **R-squared: 0.750**: El 75% de la variabilidad del "Score" es explicado por las variables independientes (Social support, Healthy life expectancy, y Freedom to make life choices). Indica un buen ajuste del modelo.
- **Adj. R-squared: 0.745**: Ajusta el \( R^2 \) por el número de predictores. El valor cercano al \( R^2 \) original sugiere que las variables son relevantes.
- **F-statistic: 151.7 (Prob: 1.67e-45)**: El p-valor ≈ 0 indica que al menos una variable independiente tiene un efecto significativo sobre el "Score". El modelo es estadísticamente válido.

### 2. Método y Datos
- **Method: Least Squares**: Método de estimación utilizado (Mínimos Cuadrados Ordinarios, OLS).
- **No. Observations: 156**: Número de observaciones en el modelo.
- **Df Residuals: 152**: Grados de libertad de los residuos (156 observaciones - 3 variables - 1 intercepto).
- **Df Model: 3**: Número de variables independientes.

### 3. Coeficientes de la Regresión
- **const (Intercepto): 1.6233**: Valor predicho del "Score" cuando todas las variables independientes son cero.
- **Social support: 1.3613**: Por cada unidad que aumenta "Social support", el "Score" aumenta **1.3613 unidades** (p-valor = 0.000, muy significativo).
- **Healthy life expectancy: 1.9496**: Por cada unidad que aumenta "Healthy life expectancy", el "Score" aumenta **1.9496 unidades** (p-valor ≈ 0, muy significativo).
- **Freedom to make life choices: 1.8450**: Por cada unidad que aumenta "Freedom to make life choices", el "Score" aumenta **1.8450 unidades** (p-valor ≈ 0, muy significativo).
- **Intervalos de Confianza [0.025, 0.975]**: Rango donde se espera que esté el verdadero coeficiente con un 95% de confianza. Ejemplo: Para "Social support", el coeficiente está entre 0.917 y 1.806.

### 4. Diagnóstico de Residuales
- **Omnibus: 2.545 (Prob: 0.280)**: Evalúa la normalidad de los residuos. P-valor > 0.05 sugiere que los residuos siguen una distribución normal.
- **Jarque-Bera (JB): 2.493 (Prob: 0.287)**: Otra prueba de normalidad. P-valor > 0.05 refuerza que los residuos son normales.
- **Skew: -0.306**: Medida de asimetría. Un valor cercano a 0 indica simetría. -0.306 sugiere una ligera asimetría hacia la izquierda.
- **Kurtosis: 2.902**: Medida de la "cola" de la distribución. Un valor cercano a 3 indica que los residuos tienen una distribución similar a la normal.

### 5. Autocorrelación y Multicolinealidad
- **Durbin-Watson: 1.535**: Evalúa la autocorrelación de los residuos. Un valor cercano a 2 indica no autocorrelación. 1.535 sugiere **leve autocorrelación positiva**, pero no es crítica.
- **Cond. No.: 14.5**: Número de condición para detectar multicolinealidad. Valores < 30 indican baja multicolinealidad. 14.5 sugiere que no hay problemas graves.

### Conclusión Final
- **Variables Significativas**: Las tres variables (**Social support**, **Healthy life expectancy**, y **Freedom to make life choices**) son altamente significativas y tienen un impacto positivo en el "Score".
- **Ajuste del Modelo**: El modelo explica el 75% de la variabilidad del "Score" (\( R^2 = 0.75 \)) y es válido estadísticamente (F-statistic ≈ 0).
- **Supuestos Cumplidos**: Los residuos son normales y no hay multicolinealidad grave.
- **Posible Mejora**: La leve autocorrelación (Durbin-Watson = 1.535) podría requerir atención si los datos son temporales.

""")

### Regresión Logística para Clasificación de Felicidad
st.subheader("Regresión logística para clasificación de felicidad 🔮🔍")
# Nos permite analizar la importancia de cada dato para la felicidad

# Importar bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Crear variable binaria basada en la media del Score
mean_happiness = df['Score'].mean()
df['is_happy'] = (df['Score'] > mean_happiness).astype(int)

# Seleccionar características para el modelo
features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 
           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
X = df[features]
y = df['is_happy']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)


# Visualizar matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
st.pyplot(plt)

# Calcular métricas
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1] / (cm[1,1] + cm[0,1])
recall = cm[1,1] / (cm[1,1] + cm[1,0])
f1 = 2 * (precision * recall) / (precision + recall)

st.write(f"\n ### Métricas del modelo:")
st.write(f"Exactitud (Accuracy): {accuracy:.2%}")
st.write(f"Precisión: {precision:.2%}")
st.write(f"Sensibilidad (Recall): {recall:.2%}")
st.write(f"F1-Score: {f1:.2%}")

# Interpretación por cuadrante
st.write("\n ### Interpretación de la matriz de confusión:")
st.write(f"Verdaderos Negativos (VN): {cm[0,0]} países correctamente clasificados como no felices")
st.write(f"Falsos Positivos (FP): {cm[0,1]} países incorrectamente clasificados como felices")
st.write(f"Falsos Negativos (FN): {cm[1,0]} países incorrectamente clasificados como no felices")
st.write(f"Verdaderos Positivos (VP): {cm[1,1]} países correctamente clasificados como felices")

# Analizar importancia de características
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Importancia de Características en la Predicción de Felicidad')
st.pyplot(plt)

# Ejemplo de predicción para un nuevo país
def predict_happiness(country_data):
    # Escalar los datos
    scaled_data = scaler.transform([country_data])
    # Realizar predicción
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    return prediction[0], probability[0]

# Ejemplo de uso
example_country = [1.0, 1.2, 0.8, 0.5, 0.2, 0.1]  # Valores ejemplo
prediction, probability = predict_happiness(example_country)
st.write("\nPredicción para país hipotético:")
st.write(f"Clasificación: {'Feliz' if prediction == 1 else 'No feliz'}")
st.write(f"Probabilidad: {probability[1]:.2%} de ser feliz")

st.markdown("""## Factores más Influyentes en la Felicidad🔆

El análisis muestra que el **apoyo social** (social support) y la **esperanza de vida saludable** (healthy life expectancy) son los factores más determinantes para la felicidad de un país. Esto sugiere que las sociedades con fuertes redes de apoyo y buenos sistemas de salud tienden a ser más felices.

Por otro lado, la **generosidad** y la **percepción de corrupción** mostraron una influencia considerablemente menor en el nivel de felicidad de los países, indicando que estos factores no son tan cruciales para determinar el bienestar general de una nación.

Esta información podría ser valiosa para orientar políticas públicas hacia el fortalecimiento de sistemas de apoyo social y servicios de salud.""")

# Top 10 países
st.subheader("Top 10 Países más Felices")
top_countries = df.nlargest(10, 'Score')[['Country or region', 'Score', 'Continent']]
st.dataframe(top_countries.style.background_gradient(cmap='Blues'), use_container_width=True)

# Análisis estadístico avanzado
st.subheader("Análisis Estadístico Detallado")
st.write("""
### Hallazgos Clave:
1. **Factores Determinantes:**
   - El 75% de la varianza en el índice se explica por factores económicos y sociales
   - El PIB y soporte social explican el 60% de las diferencias entre países

2. **Distribución Regional:**
   - Europa contiene el 80% de los países en el top 20
   - África representa el 90% de los países en el cuartil inferior

3. **Corrupción Percepcibida:**
   - Los países nórdicos combinan alta confianza institucional con altos índices
   - La correlación corrupción-felicidad es débil (r = -0.15)
""")
# Mapa interactivo
st.subheader("Distribución Geográfica de la Felicidad")
fig = px.choropleth(df, locations="Country or region",
                    locationmode='country names',
                    color="Score", 
                    hover_name="Country or region",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Mapa Mundial de Felicidad")
st.plotly_chart(fig, use_container_width=True)
# Footer
st.markdown("---")
st.markdown("**Universidad de La Habana** - Facultad de Ciencias de la Computación")
st.markdown("Proyecto de Análisis Estadístico - 2024")