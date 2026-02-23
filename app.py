import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AIQON | Command Center", layout="wide")
CORES = {'LinkedIn': '#0A66C2', 'Mailchimp': '#F97316', 'Blog': '#10B981', 'Total': '#1E293B', 'Fundo': '#F8FAFC'}

@st.cache_data
def carregar_dados_e_dax():
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')

    # ==========================================
    # 🧠 MÉTRICAS DAX: LINKEDIN
    # ==========================================
    lin['Total Engajamento'] = lin['Curtidas'] + lin['Comentários'] + lin['Shares']
    lin['Taxa Engajamento (ER)'] = np.where(lin['Seguidores'] == 0, 0, lin['Total Engajamento'] / lin['Seguidores'])
    
    def tamanho_post(texto):
        l = len(str(texto))
        if l < 500: return "Curto (Teaser)"
        elif l < 1200: return "Médio (Padrão)"
        else: return "Longo (Artigo/Deep Dive)"
    lin['Tamanho do Post'] = lin['Texto'].apply(tamanho_post)

    def tipo_conteudo(texto):
        t = str(texto).lower()
        if 'parceria' in t or 'solução' in t: return "Comercial/Venda"
        elif 'dica' in t or 'sabia que' in t or 'tendência' in t: return "Editorial/Educativo"
        else: return "News/Geral"
    lin['Tipo Conteúdo'] = lin['Texto'].apply(tipo_conteudo)

    def tag_produto(t):
        t = str(t).lower()
        if 'action1' in t: return 'Action1'
        elif 'netwrix' in t: return 'Netwrix'
        elif '42crunch' in t: return '42Crunch'
        elif 'syxsense' in t: return 'Syxsense'
        elif 'ox' in t: return 'Ox Security'
        elif 'cynet' in t: return 'Cynet'
        elif 'easy inventory' in t: return 'Easy Inventory'
        elif 'keepit' in t: return 'Keepit'
        elif 'grip' in t: return 'Grip'
        elif 'manage engine' in t: return 'Manage Engine'
        elif 'wallarm' in t: return 'Wallarm'
        else: return 'Institucional / Outros'
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)

    # ==========================================
    # 🧠 MÉTRICAS DAX: BLOG
    # ==========================================
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Título Post'] = blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.upper()
    blo['Taxa Conversão Blog'] = np.where(blo['Views'] == 0, 0, blo['Clicks'] / blo['Views'])
    
    # Recriando o Tooltip do Blog (ALL('Blog'))
    tempo_global = blo['Tempo da Página'].mean()
    blo['Diff Retencao'] = np.where(tempo_global == 0, 0, (blo['Tempo da Página'] - tempo_global) / tempo_global)
    blo['Texto_Direcao'] = np.where(blo['Diff Retencao'] > 0, "acima", "abaixo")
    # Formatando string do Tooltip
    blo['Texto Performance Blog'] = "⏳ Retenção: " + (blo['Diff Retencao'].abs() * 100).map('{:.1f}%'.format) + " " + blo['Texto_Direcao'] + " da média do blog.<br>🎯 Conversão: Gerou um total de " + blo['Clicks'].astype(str) + " cliques em produtos."

    # ==========================================
    # 🧠 MÉTRICAS DAX: MAILCHIMP
    # ==========================================
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    mai['Total Aberturas Absolutas'] = np.round(mai['Qtd Enviados'] * mai['Taxa de Abertura']).astype(int)
    mai['Total Cliques Email'] = np.round(mai['Qtd Enviados'] * mai['Clicks']).astype(int)
    mai['Total Ignorados'] = mai['Qtd Enviados'] - mai['Total Aberturas Absolutas']
    mai['Taxa de Clique por Abertura (CTOR)'] = np.where(mai['Total Aberturas Absolutas'] == 0, 0, mai['Total Cliques Email'] / mai['Total Aberturas Absolutas'])

    # Recriando o Tooltip do Mailchimp (ALLSELECTED('Mailchimp'))
    ctor_global = mai['Total Cliques Email'].sum() / mai['Total Aberturas Absolutas'].sum() if mai['Total Aberturas Absolutas'].sum() > 0 else 0
    mai['Diff CTOR'] = np.where(ctor_global == 0, 0, (mai['Taxa de Clique por Abertura (CTOR)'] - ctor_global) / ctor_global)
    mai['Texto Performance Tooltip'] = "Este e-mail performou " + (mai['Diff CTOR'] * 100).map('{:+.1f}%'.format) + " em relação à média de cliques."

    # ==========================================
    # 🌐 OVERVIEW (FULL BLAST)
    # ==========================================
    r_lin = lin.groupby(['Data da postagem', 'Tag Produto'])['Total Engajamento'].sum().reset_index().rename(columns={'Data da postagem': 'Data'})
    r_blo = blo.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index()
    r_mai = mai.groupby(['Data de Envio', 'Tag Produto'])['Total Aberturas Absolutas'].sum().reset_index().rename(columns={'Data de Envio': 'Data'})

    over = pd.merge(r_lin, r_blo, on=['Data', 'Tag Produto'], how='outer')
    over = pd.merge(over, r_mai, on=['Data', 'Tag Produto'], how='outer').fillna(0)
    over['Impacto Total'] = over['Total Engajamento'] + over['Views'] + over['Total Aberturas Absolutas']
    over = over.dropna(subset=['Data']).sort_values('Data')

    return over, lin, blo, mai

df_overview, df_linkedin, df_blog, df_mailchimp = carregar_dados_e_dax()

# ==========================================
# INTERFACE B2B
# ==========================================
st.sidebar.title("🎯 Command Center")
menu = st.sidebar.radio("Navegação:", ["🌐 Full Blast", "💼 LinkedIn", "📧 Mailchimp", "📝 Blog"])

if menu == "🌐 Full Blast":
    st.title("Evolução Full Blast")
    evo = df_overview.groupby('Data').sum(numeric_only=True).reset_index()
    
    fig = px.area(evo, x='Data', y='Impacto Total', color_discrete_sequence=[CORES['Total']], custom_data=['Total Engajamento', 'Views', 'Total Aberturas Absolutas'])
    fig.update_traces(hovertemplate="<b>Data:</b> %{x}<br><b>Impacto Total:</b> %{y}<br><br>LinkedIn: %{customdata[0]}<br>Blog: %{customdata[1]}<br>Mailchimp: %{customdata[2]}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

elif menu == "💼 LinkedIn":
    st.title("LinkedIn: Desempenho de IA & Engajamento")
    ab = df_linkedin.groupby(['Tipo Conteúdo', 'Tamanho do Post'])['Total Engajamento'].mean().reset_index()
    fig = px.bar(ab, x='Tipo Conteúdo', y='Total Engajamento', color='Tamanho do Post', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "📧 Mailchimp":
    st.title("Mailchimp: Abertura vs CTOR")
    
    # Gráfico com o Tooltip Exato do DAX
    fig = px.scatter(
        df_mailchimp, x='Taxa de Abertura', y='Taxa de Clique por Abertura (CTOR)', 
        size='Qtd Enviados', color='Tag Produto', hover_name='Título',
        custom_data=['Texto Performance Tooltip', 'Total Ignorados', 'Total Aberturas Absolutas']
    )
    # Desenhando o Tooltip personalizado do Plotly
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br><br>Taxa de Abertura: %{x:.1%}<br>CTOR: %{y:.1%}<br><br>📊 Aberturas: %{customdata[2]}<br>❌ Ignorados: %{customdata[1]}<br><br><i>%{customdata[0]}</i><extra></extra>"
    )
    fig.add_vline(x=df_mailchimp['Taxa de Abertura'].mean(), line_dash="dot", line_color="red")
    fig.add_hline(y=df_mailchimp['Taxa de Clique por Abertura (CTOR)'].mean(), line_dash="dot", line_color="red")
    fig.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
    
    st.plotly_chart(fig, use_container_width=True)

elif menu == "📝 Blog":
    st.title("Blog: Retenção e Views")
    
    # Gráfico com o Tooltip Exato do DAX
    fig = px.scatter(
        df_blog, x='Views', y='Tempo da Página', size='Clicks', color='Tag Produto', hover_name='Título Post',
        custom_data=['Texto Performance Blog', 'Taxa Conversão Blog']
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Taxa Conversão: %{customdata[1]:.1%}<br>Views: %{x}<br>Tempo: %{y}s<br><br>%{customdata[0]}<extra></extra>"
    )
    fig.add_hline(y=df_blog['Tempo da Página'].mean(), line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
