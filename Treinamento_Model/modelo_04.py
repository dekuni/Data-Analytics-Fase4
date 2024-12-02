import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from joblib import load
import plotly.graph_objects as go
import plotly.express as px

# Configuração da página
st.set_page_config(layout="wide")
st.title("Dashboard de Previsão do Preço do Petróleo")

# Caminhos para os arquivos
model_path = r"C:\Users\da\Desktop\Streamlit\prophet_model.pkl"
forecast_path = r"C:\Users\da\Desktop\Streamlit\forecast.csv"
base_path = r"C:\Users\da\Desktop\Streamlit\BASE.csv"

# Eventos históricos para adicionar aos gráficos
eventos = [
    {"Data": "1990-08-02", "descricao": "Início da Guerra do Golfo"},
    {"Data": "1991-02-28", "descricao": "Fim da Guerra do Golfo"},
    {"Data": "2020-05-01", "descricao": "Acordo de produção OPEP"},
    {"Data": "2020-09-01", "descricao": "Tensão política no Oriente Médio"},
    {"Data": "2008-07-11", "descricao": "Preço recorde do petróleo - Crise financeira"},
    {"Data": "2020-03-01", "descricao": "Impacto inicial da pandemia"},
    {"Data": "2014-06-20", "descricao": "Queda de preços devido ao excesso de oferta"}
]

# Criar abas
tab1, tab2, tab3, tab4 = st.tabs(["Introdução", "Gráficos e Análises", "Modelo de Machine Learning", "Conclusão"])

# Aba 1: Introdução
with tab1:
    st.header("Introdução")
    st.write("""
    Este dashboard apresenta insights sobre a variação do preço do petróleo e uma previsão baseada em modelos de Machine Learning. 
    Ele combina análises interativas e previsões para auxiliar na tomada de decisões em mercados voláteis.
    """)

# Aba 2: Gráficos e Análises
with tab2:
    st.header("Gráficos e Análises")

    try:
        # Carregar os dados
        base = pd.read_csv(base_path, sep=';')
        base['y'] = base['y'].str.replace(',', '.').astype(float)
        base['ds'] = pd.to_datetime(base['ds'])

        # Gráfico 1: Série Temporal com Eventos
        st.subheader("Série Temporal com Eventos Econômicos")
        fig_serie = go.Figure()
        fig_serie.add_trace(go.Scatter(x=base["ds"], y=base["y"], mode="lines+markers", name="Preço", line=dict(color="blue")))

        # Adicionar os eventos ao gráfico
        for evento in eventos:
            if evento["Data"] in base["ds"].astype(str).values:
                fig_serie.add_annotation(
                    x=evento["Data"],
                    y=base.loc[base["ds"] == evento["Data"], "y"].values[0],
                    text=evento["descricao"],
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    bgcolor="yellow",
                    bordercolor="black",
                    font=dict(size=10),
                )
        fig_serie.update_layout(title="Série Temporal do Preço do Petróleo", xaxis_title="Data", yaxis_title="Preço (USD)", template="plotly_white")
        st.plotly_chart(fig_serie, use_container_width=True)

        st.write("""
        Este gráfico destaca eventos econômicos e geopolíticos que impactaram diretamente os preços do petróleo. 
        Crises financeiras, conflitos e pandemias são fatores decisivos na volatilidade do mercado de energia.
        """)

        # Gráfico 2: Evolução Mensal
        st.subheader("Evolução do Preço Médio Mensal com Destaques")
        base["Mes_Ano"] = base["ds"].dt.to_period("M").dt.to_timestamp()
        df_mensal = base.groupby("Mes_Ano")["y"].mean().reset_index()

        top = df_mensal.nlargest(3, "y")
        bottom = df_mensal.nsmallest(3, "y")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_mensal["Mes_Ano"], y=df_mensal["y"], mode="lines", name="Preço Médio Mensal", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=top["Mes_Ano"], y=top["y"], mode="markers", marker=dict(size=10, color="red"), name="Altas"))
        fig.add_trace(go.Scatter(x=bottom["Mes_Ano"], y=bottom["y"], mode="markers", marker=dict(size=10, color="green"), name="Baixas"))
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        Este gráfico destaca os meses com os preços mais altos e mais baixos do petróleo. 
        As maiores altas geralmente coincidem com crises e tensões globais, enquanto as baixas refletem excesso de oferta ou queda na demanda.
        """)

        # Gráfico 3: Média Móvel
        st.subheader("Projeção com Média Móvel (30 dias)")
        base["Média Móvel (30 dias)"] = base["y"].rolling(window=30).mean()
        fig_projecao = px.line(base, x="ds", y=["y", "Média Móvel (30 dias)"], title="Projeção - Média Móvel (30 dias)")
        st.plotly_chart(fig_projecao, use_container_width=True)

        st.write("""
        A média móvel suaviza as flutuações diárias, facilitando a identificação de tendências de longo prazo no mercado de petróleo.
        """)

    except Exception as e:
        st.error(f"Erro ao processar os gráficos: {e}")

# Aba 3: Modelo de Machine Learning
with tab3:
    st.header("Modelo de Machine Learning")

    try:
        # Carregar o modelo Prophet
        model = load(model_path)

        # Carregar previsões
        forecast = pd.read_csv(forecast_path)
        st.write("Previsões carregadas:")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Gráfico interativo com previsões
        st.subheader("Gráfico de Previsões")
        forecast_plot = plot_plotly(model, forecast)
        forecast_plot.update_layout(width=800, height=600)
        st.plotly_chart(forecast_plot, use_container_width=True)

        # Métricas do modelo
        st.write("**Métricas do Modelo:**")
        st.write("""
        - **Erro Médio Absoluto (MAE)**: 6.94  
        - **Erro Médio Quadrático (MSE)**: 118.32  
        - **Raiz do Erro Médio Quadrático (RMSE)**: 10.88  
        - **R²**: 0.89  
        """)
        st.write("""
        O modelo Prophet apresentou um desempenho consistente, captando bem as tendências de longo prazo, 
        embora ainda enfrente desafios em momentos de alta volatilidade.
        """)

    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou previsões: {e}")

# Aba 4: Conclusão
with tab4:
    st.header("Conclusão")
    st.write("""
    Este dashboard mostrou como dados históricos e previsões baseadas em Machine Learning podem fornecer 
    insights valiosos para mercados voláteis como o de petróleo.  
    Eventos geopolíticos e crises globais desempenham um papel crucial na formação dos preços, enquanto 
    o modelo de previsão pode ajudar a antecipar tendências, auxiliando na tomada de decisão estratégica.
    """)
