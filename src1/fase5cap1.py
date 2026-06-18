
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
import streamlit as st
import plotly.express as px



df = pd.read_csv("farmtech_data.csv")
df.head()

df.info()

duplicates = df.duplicated().sum()
nulos= df.isnull().sum()
print("numero de dados nulos", nulos)
print("numero de dados duplicados", duplicates)
print("numero de dados duplicados", duplicates)
#primeiro é necessario verificar se existem duplicatas no code
#verificar a presença de outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.title("boxplot, para detectar outliers")
plt.xticks(rotation=45)
plt.show()

#verificação de valores maximos
df.max()

#verificação dos valores minimos
df.min()

df.hist(figsize=(12,10))

plt.suptitle("Distribuição dos Dados", fontsize=16)

plt.show()


#Correlação entre as features numéricas
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#aqui na parte de cima faz considerar como numéricas todas as colunas que sejam inteiras ou decimais
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=numerics).corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação")
plt.show()


X = df.drop("produtividade_kg_ha", axis=1)
y = df["produtividade_kg_ha"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr) ** 0.5)
print("R²:", r2_score(y_test, y_pred_lr))

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

print("Decision Tree")
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("RMSE:", mean_squared_error(y_test, y_pred_dt) ** 0.5)
print("R²:", r2_score(y_test, y_pred_dt))

#Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("random forest")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RMSE:", mean_squared_error(y_test, y_pred_rf) ** 0.5)
print("R²:", r2_score(y_test, y_pred_rf))

# SVR Linear
# célula 1
svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train_scaled, y_train)
y_pred_svr_linear = svr_linear.predict(X_test_scaled)

print("SVR Linear")
print("MAE:", mean_absolute_error(y_test, y_pred_svr_linear))
print("RMSE:", mean_squared_error(y_test, y_pred_svr_linear) ** 0.5)
print("R²:", r2_score(y_test, y_pred_svr_linear))

# SVR Poly
svr_poly = SVR(kernel='poly')
svr_poly.fit(X_train_scaled, y_train)
y_pred_svr_poly = svr_poly.predict(X_test_scaled)

print("SVR Poly")
print("MAE:", mean_absolute_error(y_test, y_pred_svr_poly))
print("RMSE:", mean_squared_error(y_test, y_pred_svr_poly) ** 0.5)
print("R²:", r2_score(y_test, y_pred_svr_poly))

# SVR RBF
svr_rbf = SVR(kernel='rbf')
svr_rbf.fit(X_train_scaled, y_train)
y_pred_svr_rbf = svr_rbf.predict(X_test_scaled)

print("SVR RBF")
print("MAE:", mean_absolute_error(y_test, y_pred_svr_rbf))
print("RMSE:", mean_squared_error(y_test, y_pred_svr_rbf) ** 0.5)
print("R²:", r2_score(y_test, y_pred_svr_rbf))

#DASHBOARD
st.title("🌱 FarmTech Analytics")
st.write("Análise de Dados e Predição Inteligente da Produtividade Rural")

umidade_media = df["umidade_solo"].mean()
ph_media = df["pH_solo"].mean()
prod_media = df["produtividade_kg_ha"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("média da umidade",umidade_media)
col2.metric("ph médio do solo", f"{ph_media:.1f}")
col3.metric("produtividade média Kg/Ha",f"{prod_media:.0f}kg/ha")


#GRAFICO 1

# matriz de correlação
corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlação entre Variáveis"
)

fig.update_layout(
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

#GRAFICO 2 UMIDADE X PRODUTIVIDADE

fig = px.scatter(
    df,
    x="umidade_solo",
    y="produtividade_kg_ha",
    title="Relação entre Umidade do Solo e Produtividade",
    trendline="ols"  # adiciona linha de tendência
)

fig.update_layout(
    xaxis_title="Umidade do Solo (%)",
    yaxis_title="Produtividade (kg/ha)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

#GRAFICO 3 PH X PRODUTIVIDADE

fig = px.scatter(
    df,
    x="pH_solo",
    y="produtividade_kg_ha",
    title="Relação entre PH do Solo e Produtividade",
    trendline="ols"  # adiciona linha de tendência
)

fig.update_layout(
    xaxis_title="PH do solo (%)",
    yaxis_title="Produtividade (kg/ha)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

#GRAFICO 4 produtividade por faixa de umidade
df["faixa_umidade"] = pd.cut(
    df["umidade_solo"],
    bins=5
).astype(str)

fig = px.box(
    df,
    x="faixa_umidade",
    y="produtividade_kg_ha",
    title="Produtividade por Faixa de Umidade"
)

fig.update_layout(
    xaxis_title="Faixa de Umidade do Solo",
    yaxis_title="Produtividade (kg/ha)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.title("PREVER DADOS DE PRODUTIVIDADE")
st.write("insira dados para que nossa machine learning treinada possa prever a pordutividade")
umidade_solo = st.number_input("Umidade do solo (%)")
pH_solo = st.number_input("pH do solo")
temperatura = st.number_input("Temperatura (°C)")
chuva_mm = st.number_input("Chuva (mm)")
nitrogenio = st.number_input("Nitrogênio")
fosforo = st.number_input("Fósforo")
potassio = st.number_input("Potássio")
irrigacao_l = st.number_input("Irrigação (L)")
fertilizante_kg = st.number_input("Fertilizante (kg)")

if st.button("Prever produtividade"):

    dados_novos = pd.DataFrame({
        "umidade_solo": [umidade_solo],
        "pH_solo": [pH_solo],
        "temperatura": [temperatura],
        "chuva_mm": [chuva_mm],
        "nitrogenio": [nitrogenio],
        "fosforo": [fosforo],
        "potassio": [potassio],
        "irrigacao_l": [irrigacao_l],
        "fertilizante_kg": [fertilizante_kg]
    })

    dados_scaled = scaler.transform(dados_novos)

    produtividade_prevista = lr.predict(dados_scaled)

    st.success(f"Produtividade prevista: {produtividade_prevista[0]:.2f} kg/ha")

    if umidade_solo < 30:
        st.error("Recomendação: aumentar a irrigação.")
    elif pH_solo < 5.5:
        st.warning("Recomendação: corrigir a acidez do solo.")
    elif nitrogenio < 30 or fosforo < 20 or potassio < 20:
        st.warning("Recomendação: revisar fertilização do solo.")
    elif produtividade_prevista[0] < 2500:
        st.warning("Recomendação: revisar manejo agrícola.")
    else:
        st.success("Condições adequadas para boa produtividade.")