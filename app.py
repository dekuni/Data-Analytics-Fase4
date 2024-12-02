import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from joblib import load
import plotly.graph_objects as go
import plotly.express as px

# Configuração da página
st.set_page_config(layout="centered")
st.title("Análise do preço do Petróleo")


# Caminhos para os arquivos
model_path = "prophet_model.pkl"
forecast_path = "forecast.csv"
base_path = "BASE.csv"

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
    st.divider()
    st.write("""
A análise dos preços do petróleo é um dos temas mais críticos e dinâmicos na economia global, dado o seu impacto direto em múltiplos setores, como a indústria energética, o transporte, a agricultura, a manufatura e até as políticas públicas. O petróleo, como um dos principais recursos naturais do mundo, exerce uma influência profunda sobre os preços de bens e serviços, afetando não apenas o mercado financeiro, mas também as políticas econômicas e as relações internacionais. Sua importância estratégica transcende fronteiras, tornando-se um elemento chave na geopolítica global e nas decisões econômicas de nações e corporações.

A volatilidade dos preços do petróleo é impulsionada por uma série de fatores interconectados e muitas vezes imprevisíveis. A interação de crises econômicas globais, como recessões e falências de grandes instituições financeiras, com eventos geopolíticos, como guerras e disputas comerciais, cria um cenário altamente instável. Além disso, a mudança na demanda global, impulsionada por fatores como o crescimento econômico de potências emergentes ou a implementação de políticas ambientais mais rigorosas, também impacta diretamente o valor do petróleo. Organizações como a OPEP (Organização dos Países Exportadores de Petróleo) desempenham um papel fundamental nesse processo, já que suas decisões de produção afetam a oferta e, consequentemente, o preço do barril.

Compreender e prever essas flutuações é essencial para qualquer análise econômica e estratégica. Identificar tendências no comportamento do preço do petróleo, antecipar possíveis crises e riscos, e formular estratégias adaptativas são passos fundamentais para empresas, governos e investidores. No entanto, devido à complexidade e ao comportamento altamente volátil desse mercado, é necessária uma abordagem analítica avançada para capturar os padrões e nuances do mercado de petróleo ao longo do tempo.

Este trabalho propõe um dashboard interativo que integra análises históricas com previsões baseadas em técnicas avançadas de Machine Learning. Utilizando modelos estatísticos como o Prophet, o dashboard não apenas visualiza os padrões passados e as sazonalidades históricas dos preços do petróleo, mas também oferece projeções futuras, permitindo uma antecipação dos movimentos do mercado. A integração de eventos históricos marcantes, como a crise financeira de 2008, a queda dos preços devido à superoferta de petróleo em 2014 e a pandemia de COVID-19, proporciona uma análise contextualizada que enriquece as previsões. Com essas ferramentas, o dashboard se torna um recurso imprescindível para gestores, analistas financeiros e tomadores de decisão, oferecendo um suporte robusto para a tomada de decisões estratégicas em um mercado de grande volatilidade e incerteza.""")

    st.divider()
    st.write("**Alunos:**\n\n Daniel Guimarães Carvalho,\n\n danielccl.gui.dc@gmail.com\n\n ALEX FELIPE PINHATARI RODRIGUES FILHO, \n\n ALEX.FRODRIGUES71@GMAIL.COM\n\n GABRIEL INACIO VIEIRA MARTINS,\n\n gabriel_inacio54@outlook.com")

# Aba 2: Gráficos e Análises
with tab2:
    st.header("Gráficos e Análises")

    try:
        # Carregar os dados
        base = pd.read_csv(base_path, sep=';')
        base['y'] = base['y'].str.replace(',', '.').astype(float)
        base['ds'] = pd.to_datetime(base['ds'])

        # Gráfico 1: Série Temporal com Eventos Clicáveis
        st.subheader("Série Temporal com Eventos Econômicos")
        fig_serie = go.Figure()
        fig_serie.add_trace(go.Scatter(x=base["ds"], y=base["y"], mode="lines+markers", name="Preço", line=dict(color="blue")))

        for evento in eventos:
            if evento["Data"] in base["ds"].astype(str).values:
                fig_serie.add_trace(
                    go.Scatter(
                        x=[evento["Data"]],
                        y=[base.loc[base["ds"] == evento["Data"], "y"].values[0]],
                        mode="markers+text",
                        name=evento["descricao"],
                        text=evento["descricao"],
                        textposition="top center",
                        marker=dict(size=10, color="lightblue", symbol="circle"),
                        hovertemplate=f"<b>Evento:</b> {evento['descricao']}<br><b>Data:</b> {evento['Data']}<br><b>Preço:</b> $%{{y:.2f}}"
                    )
                )
        fig_serie.update_layout(
            title="Série Temporal do Preço do Petróleo com Eventos",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            template="plotly_white",
            showlegend=True
        )
        st.plotly_chart(fig_serie, use_container_width=True)

        st.write("""
        Este gráfico apresenta a evolução do preço do petróleo ao longo do tempo, destacando eventos econômicos e geopolíticos que impactaram 
        significativamente o mercado. Momentos como a crise financeira global de 2008, o início da pandemia de COVID-19 em 2020 e conflitos 
        geopolíticos aparecem diretamente no gráfico como marcos históricos que influenciaram as flutuações. A interatividade permite uma análise 
        detalhada desses eventos e como eles coincidem com mudanças bruscas no preço, facilitando a compreensão de como fatores externos afetam o mercado.
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
        O gráfico de evolução do preço médio mensal destaca os meses com preços mais altos e mais baixos ao longo do período analisado. As maiores 
        altas costumam refletir tensões globais, crises ou eventos que impactaram a oferta e a demanda. Por outro lado, os períodos de menor preço 
        estão associados a uma oferta abundante, reduções na demanda ou situações de estabilidade econômica. Este gráfico oferece uma visão clara 
        das dinâmicas sazonais e globais que moldam o comportamento do mercado ao longo do tempo.
        """)

        # Gráfico 3: Média Móvel
        st.subheader("Projeção com Média Móvel (30 dias)")
        base["Média Móvel (30 dias)"] = base["y"].rolling(window=30).mean()
        fig_projecao = px.line(base, x="ds", y=["y", "Média Móvel (30 dias)"], title="Projeção - Média Móvel (30 dias)")
        st.plotly_chart(fig_projecao, use_container_width=True)

        st.write("""
        A média móvel de 30 dias é uma ferramenta essencial para identificar tendências de longo prazo. Ao suavizar as variações diárias no preço 
        do petróleo, ela elimina os "ruídos" do mercado e permite uma análise mais clara de padrões consistentes. Com essa projeção, é possível 
        observar momentos de alta e baixa ao longo do tempo, fornecendo uma base sólida para análises preditivas e tomadas de decisão no mercado.
        """)

        # Gráfico 4: Volatilidade Anual
        st.subheader("Volatilidade Anual do Preço do Petróleo")
        base = base.sort_values("ds")
        base["Retorno Diário (%)"] = base["y"].pct_change() * 100
        base["Ano"] = base["ds"].dt.year

        volatilidade_anual = base.groupby("Ano")["Retorno Diário (%)"].std() * (252**0.5)

        ano_max_volatilidade = volatilidade_anual.idxmax()
        ano_min_volatilidade = volatilidade_anual.idxmin()

        cores = ["orange" if ano not in [ano_max_volatilidade, ano_min_volatilidade]
                 else "red" if ano == ano_max_volatilidade
                 else "green"
                 for ano in volatilidade_anual.index]

        fig_volatilidade = go.Figure()
        fig_volatilidade.add_trace(
            go.Bar(
                x=volatilidade_anual.index,
                y=volatilidade_anual.values,
                name="Volatilidade Anual",
                marker_color=cores,
                hovertemplate="Ano: %{x}<br>Volatilidade: %{y:.2f}%"
            )
        )
        fig_volatilidade.update_layout(
            title="Volatilidade Anual do Preço do Petróleo",
            xaxis_title="Ano",
            yaxis_title="Volatilidade (%)",
            template="plotly_white",
            title_font=dict(size=20),
            xaxis=dict(tickmode="linear", tick0=volatilidade_anual.index.min(), dtick=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig_volatilidade.add_annotation(
            x=ano_max_volatilidade,
            y=volatilidade_anual.max(),
            text=f"Maior Volatilidade: {volatilidade_anual.max():.2f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-60,
            font=dict(color="red", size=12)
        )
        fig_volatilidade.add_annotation(
            x=ano_min_volatilidade,
            y=volatilidade_anual.min(),
            text=f"Menor Volatilidade: {volatilidade_anual.min():.2f}%",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-60,
            font=dict(color="green", size=12)
        )
        st.plotly_chart(fig_volatilidade, use_container_width=True)

        st.write("""
        A volatilidade anual do preço do petróleo é uma métrica crucial para avaliar os riscos associados ao mercado. Este gráfico identifica 
        os anos mais voláteis, que coincidem com grandes choques, como crises econômicas, tensões geopolíticas ou mudanças abruptas na demanda 
        global. Por outro lado, os períodos de menor volatilidade refletem estabilidade e previsibilidade no mercado. A análise desses padrões 
        oferece insights profundos sobre como eventos globais impactam os preços e a confiança dos investidores.
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
        st.write("O modelo de previsão apresentou resultados bastante positivos, tanto nas métricas de avaliação quanto na análise visual do gráfico. O Erro Médio Absoluto (MAE), de 6,94, mostra que, em média, as previsões do modelo diferem dos valores reais por uma margem relativamente pequena. Isso significa que ele conseguiu fazer previsões próximas dos valores reais em muitos casos.\n\nO Erro Quadrático Médio (MSE) foi de 118,32, e, quando tiramos a raiz quadrada dele (RMSE), obtemos um valor de 10,88, que nos dá uma noção do desvio médio. Esses resultados são bons, considerando a escala dos dados.\n\nO R², que ficou em 0,89, é outro ponto importante. Isso quer dizer que o modelo foi capaz de explicar 89% das variações dos dados reais, o que é um excelente indicador de que ele está captando bem os padrões dessa série temporal.\n\nOlhando para o gráfico, vemos que as previsões acompanham bem os valores reais (a linha preta), especialmente nas tendências de longo prazo. O modelo conseguiu identificar os comportamentos gerais da série e apresentou boas estimativas. As áreas sombreadas no gráfico, que mostram a incerteza das previsões, permanecem relativamente estreitas na maior parte do tempo, o que passa confiança nos resultados. Ainda assim, há momentos de maior volatilidade nos dados reais, como picos e quedas acentuadas, que o modelo teve um pouco mais de dificuldade para capturar com precisão — algo esperado em séries temporais complexas.\n\nNo geral, o desempenho do modelo foi muito bom. Ele é confiável e pode ser uma ferramenta útil para ajudar a tomar decisões baseadas nas previsões. Com ajustes futuros, talvez seja possível melhorar ainda mais a precisão em momentos mais instáveis, mas os resultados atuais já mostram um modelo sólido e bem ajustado.")

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
    st.divider()
    st.write("""
Este estudo evidenciou o enorme potencial de integrar análises históricas detalhadas e modelos de Machine Learning para desvendar padrões ocultos e fornecer insights estratégicos sobre um dos mercados mais dinâmicos e imprevisíveis do mundo. A análise revelou que fatores geopolíticos, como guerras e tensões regionais, assim como crises econômicas globais e eventos disruptivos, como a pandemia de COVID-19, exercem um impacto significativo nos preços do petróleo. Incorporar esses eventos no contexto analítico permite compreender como forças externas influenciam a volatilidade e a formação de preços, destacando a importância de decisões estratégicas bem fundamentadas em cenários de alta incerteza.

O modelo Prophet, com métricas sólidas como um MAE de 6,94 e um R² de 0,89, demonstrou sua eficácia em capturar tendências de longo prazo, oferecendo previsões confiáveis para auxiliar gestores e investidores a planejar ações e mitigar riscos. Ele se mostrou particularmente eficaz em prever comportamentos gerais e padrões sazonais, mesmo em um mercado conhecido por sua complexidade e alta volatilidade.

Os gráficos interativos fornecidos, como o de volatilidade anual, evolução de preços médios mensais e projeções com médias móveis, ampliaram a compreensão das dinâmicas sazonais e dos impactos de eventos globais sobre os preços. Esses gráficos não apenas explicitaram as flutuações históricas, mas também facilitaram a identificação de períodos de estabilidade e instabilidade, ajudando a embasar estratégias de longo prazo.

Embora desafios persistam em momentos de maior volatilidade, quando os dados apresentam picos ou quedas abruptas, os resultados deste estudo reforçam o valor inestimável das tecnologias analíticas avançadas na gestão de riscos e na construção de estratégias robustas. Este dashboard não é apenas uma ferramenta informativa, mas também um recurso estratégico que exemplifica o poder da inteligência de dados na transformação de informações complexas em decisões assertivas e orientadas por evidências.

Com aprimoramentos contínuos, como a incorporação de mais variáveis externas e o refinamento dos modelos, este tipo de abordagem pode evoluir para oferecer ainda mais precisão e abrangência, consolidando seu papel como um pilar fundamental na análise de mercados voláteis como o de petróleo.""")

    st.divider()
    st.header("Fontes:")
    st.write("BANCO DE DADOS IPEA: http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view")
    st.write("Evolução dos preços internacionais: https://www.ibp.org.br/observatorio-do-setor/snapshots/evolucao-dos-precos-internacionais-do-petroleo-e-projecoes-para-2025/")
    st.write("Biblioteca para criação do site: https://streamlit.io")