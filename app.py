import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuração ---
st.set_page_config(page_title="AIQON | Command Center", layout="wide")
CORES = {'LinkedIn': '#0A66C2', 'Mailchimp': '#F97316', 'Blog': '#10B981', 'Total': '#1E293B', 'Fundo': '#F8FAFC'}

# --- Carregando Dados (Leitura Rápida do GitHub) ---
@st.cache_data
def carregar_dados():
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')
    
    # Criando o Overview (Merge)
    r_lin = lin.groupby(['Data', 'Tag Produto'])['Total Engajamento'].sum().reset_index()
    r_blo = blo.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index()
    r_mai = mai.groupby(['Data', 'Tag Produto'])['Aberturas Absolutas'].sum().reset_index()

    over = pd.merge(r_lin, r_blo, on=['Data', 'Tag Produto'], how='outer')
    over = pd.merge(over, r_mai, on=['Data', 'Tag Produto'], how='outer').fillna(0)
    over['Impacto Total'] = over['Total Engajamento'] + over['Views'] + over['Aberturas Absolutas']
    over = over.dropna(subset=['Data']).sort_values('Data')
    
    return over, lin, blo, mai

df_overview, df_linkedin, df_blog, df_mailchimp = carregar_dados()

# --- Navegação ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
st.sidebar.title("Comitê de Marketing")
menu = st.sidebar.radio("Módulo:", ["🌐 Visão 'Full Blast'", "💼 Testes A/B (LinkedIn)", "📝 SEO & Retenção (Blog)", "📧 Conversão de Canais"])

st.sidebar.markdown("---")
st.sidebar.caption("💡 Atualizado via Google Colab Pipeline")

# --- Telas ---
if menu == "🌐 Visão 'Full Blast'":
    st.title("Validação de Estratégia 'Full Blast'")
    st.markdown("Monitoramento do 'bombardeio' de notícias quentes vs atemporais (Conforme diretriz do Thiago/Maikon).")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Interações Totais da Marca", f"{int(df_overview['Impacto Total'].sum()):,}")
    c2.metric("Pico Diário (Notícia Quente)", f"{int(df_overview.groupby('Data')['Impacto Total'].sum().max()):,}")
    c3.metric("Aberturas de Newsletter", f"{int(df_overview['Aberturas Absolutas'].sum()):,}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Onda de Engajamento")
        evo = df_overview.groupby('Data').sum(numeric_only=True).reset_index()
        fig = px.area(evo, x='Data', y='Impacto Total', color_discrete_sequence=[CORES['Total']], hover_data={'Total Engajamento':True, 'Views':True, 'Aberturas Absolutas':True})
        fig.update_traces(hovertemplate="<b>Data:</b> %{x}<br><b>Barulho Total:</b> %{y}<br>Social: %{customdata[0]}<br>Blog: %{customdata[1]}<br>Mailchimp: %{customdata[2]}")
        fig.add_hline(y=evo['Impacto Total'].mean(), line_dash="dash", line_color="red", annotation_text="Média de Tração")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tração por Produto")
        rank = df_overview.groupby('Tag Produto')['Impacto Total'].sum().reset_index().sort_values('Impacto Total')
        fig2 = px.bar(rank[rank['Tag Produto'] != 'Institucional / Outros'], y='Tag Produto', x='Impacto Total', orientation='h', color_discrete_sequence=[CORES['LinkedIn']])
        st.plotly_chart(fig2, use_container_width=True)

elif menu == "💼 Testes A/B (LinkedIn)":
    st.title("Laboratório de IA: Copywriting")
    st.markdown("Identificando a 'receitinha de bolo' para a conversão de ICPs.")
    
    ab = df_linkedin.groupby(['Tipo Conteúdo', 'Tamanho Post'])['Total Engajamento'].mean().reset_index()
    fig = px.bar(ab, x='Tipo Conteúdo', y='Total Engajamento', color='Tamanho Post', barmode='group', title="O que retém mais a atenção?")
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_linkedin[['Data', 'Tag Produto', 'Tipo Conteúdo', 'Tamanho Post', 'Total Engajamento']])

elif menu == "📝 SEO & Retenção (Blog)":
    st.title("Indexação Orgânica (SEO)")
    st.markdown("Medindo a retenção real do conteúdo educativo gerado pela IA, fugindo do tráfego pago (Diretriz do Flavio).")
    
    fig = px.scatter(df_blog, x='Views', y='Tempo da Página', size='Clicks', color='Tag Produto', hover_name='Título Limpo')
    fig.add_hline(y=df_blog['Tempo da Página'].mean(), line_dash="dot", line_color="red", annotation_text="Tempo Médio Global")
    fig.add_vline(x=df_blog['Views'].mean(), line_dash="dot", line_color="red", annotation_text="Média de Acessos")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_blog[['Data', 'Título Limpo', 'Views', 'Tempo da Página', 'Taxa Conversão']])

elif menu == "📧 Conversão de Canais":
    st.title("Mailchimp: Interesse vs. Intenção")
    st.markdown("Mapeamento do comportamento das revendas e MSPs.")
    
    fig = px.scatter(df_mailchimp, x='Taxa de Abertura', y='CTOR', size='Qtd Enviados', color='Tag Produto', hover_name='Título')
    fig.add_vline(x=df_mailchimp['Taxa de Abertura'].mean(), line_dash="dot", line_color="red")
    fig.add_hline(y=df_mailchimp['CTOR'].mean(), line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
