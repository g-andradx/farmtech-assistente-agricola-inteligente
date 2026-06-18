# FarmTech Solutions – Assistente Agrícola Inteligente

## Integrantes

* Gustavo Andrade
* (Adicionar demais integrantes, se houver)

## Descrição do Projeto

O projeto FarmTech Solutions tem como objetivo aplicar técnicas de Inteligência Artificial e Ciência de Dados no contexto do agronegócio, utilizando dados agrícolas para prever a produtividade das culturas e auxiliar na tomada de decisão dos gestores rurais.

A solução integra conceitos de Machine Learning, banco de dados e visualização de informações por meio de um dashboard interativo desenvolvido em Streamlit.

## Objetivo

Desenvolver um Assistente Agrícola Inteligente capaz de:

* Armazenar dados agrícolas coletados por sensores.
* Realizar análises sobre as condições do solo e do ambiente.
* Prever a produtividade agrícola utilizando modelos de regressão.
* Apresentar os resultados em um dashboard interativo.

## Tecnologias Utilizadas

### Linguagem

* Python

### Bibliotecas

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

### Banco de Dados

* SQLite / SQL

## Estrutura do Projeto

```text
FarmTech-Solutions/
│
├── data/
│   └── dados_agricolas.csv
│
├── notebooks/
│   └── analise_dados.ipynb
│
├── src/
│   ├── modelo_regressao.py
│   ├── dashboard.py
│   └── banco_dados.py
│
├── imagens/
│   └── prints_dashboard
│
├── README.md
└── requirements.txt
```

## Base de Dados

A base de dados contém informações agrícolas utilizadas para treinamento do modelo:

* Umidade do solo
* pH do solo
* Temperatura
* Chuva
* Nitrogênio
* Fósforo
* Potássio
* Irrigação
* Fertilizante
* Produtividade

Variável alvo:

```text
produtividade_kg_ha
```

## Machine Learning

Foi utilizado o algoritmo:

```text
Linear Regression
```

O modelo foi treinado para prever a produtividade agrícola com base nas características do solo e do ambiente.

### Variáveis Utilizadas

* umidade_solo
* pH_solo
* temperatura
* chuva_mm
* nitrogenio
* fosforo
* potassio
* irrigacao_l
* fertilizante_kg

### Variável Prevista

```text
produtividade_kg_ha
```

## Métricas de Avaliação

O modelo foi avaliado utilizando:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

## Dashboard

O dashboard desenvolvido em Streamlit apresenta:

* KPIs agrícolas
* Gráficos de correlação
* Distribuição dos dados
* Previsão de produtividade
* Recomendações de manejo agrícola

## Funcionalidades

### Previsão de Produtividade

O usuário informa:

* Umidade do solo
* pH
* Temperatura
* Chuva
* Nitrogênio
* Fósforo
* Potássio
* Irrigação
* Fertilizante

O sistema retorna:

* Produtividade prevista (kg/ha)
* Recomendações agrícolas

### Recomendações

Exemplos:

* Aumentar irrigação
* Corrigir acidez do solo
* Revisar fertilização
* Manter condições atuais

## Como Executar

Instalar dependências:

```bash
pip install -r requirements.txt
```

Executar dashboard:

```bash
streamlit run dashboard.py
```

## Resultados

O projeto demonstrou a aplicação prática de Inteligência Artificial no agronegócio, permitindo prever a produtividade agrícola e apoiar decisões relacionadas à irrigação e manejo do solo.

## Vídeo Demonstrativo

Inserir link do vídeo do YouTube aqui.

## GitHub

Inserir link do repositório aqui.
