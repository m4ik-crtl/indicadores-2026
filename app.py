import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AIQON | Command Center", layout="wide")
CORES = {'LinkedIn': '#0A66C2', 'Mailchimp': '#F97316', 'Blog': '#10B981', 'Total': '#1E293B', 'Fundo': '#F8FAFC'}

@st.cache_data
def processar_dados_dax():
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')

    # --- FUNÇÕES AUXILIARES ---
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

    # ==========================================
    # 🧠 MAILCHIMP (Agrupado por Título)
    # ==========================================
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    # Transforma taxa em número absoluto para poder somar as duplicadas
    mai['Aberturas'] = np.round(mai['Qtd Enviados'] * mai['Taxa de Abertura']).astype(int)
    mai['Cliques_Abs'] = np.round(mai['Qtd Enviados'] * mai['Clicks']).astype(int)
    
    # AGRUPANDO TÍTULOS IGUAIS
    mai_grp = mai.groupby(['Título', 'Data de Envio', 'Tag Produto']).agg({
        'Qtd Enviados': 'sum',
        'Aberturas': 'sum',
        'Cliques_Abs': 'sum'
    }).reset_index()

    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados']==0, 0, mai_grp['Aberturas'] / mai_grp['Qtd Enviados'])
    mai_grp['CTOR'] = np.where(mai_grp['Aberturas']==0, 0, mai_grp['Cliques_Abs'] / mai_grp['Aberturas'])
    mai_grp['Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas']

    # DAX Tooltip
    ctor_global = mai_grp['Cliques_Abs'].sum() / mai_grp['Aberturas'].sum() if mai_grp['Aberturas'].sum() > 0 else 0
    mai_grp['Diff CTOR'] = np.where(ctor_global==0, 0, (mai_grp['CTOR'] - ctor_global) / ctor_global)
    mai_grp['Tooltip'] = "Este e-mail performou " + (mai_grp['Diff CTOR']*100).map('{:+.1f}%'.format) + " em relação à média global."

    # ==========================================
    # 🧠 BLOG (Agrupado por URL)
    # ==========================================
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Título Post'] = blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.upper()
    
    # AGRUPANDO URLS IGUAIS
    blo_grp = blo.groupby(['URL', 'Título Post', 'Data', 'Tag Produto']).agg({
        'Views': 'sum',
        'Clicks': 'sum',
        'Tempo da Página': 'mean' # Média de tempo para quem leu várias vezes
    }).reset_index()

    blo_grp['Taxa Conversão'] = np.where(blo_grp['Views']==0, 0, blo_grp['Clicks'] / blo_grp['Views'])
    
    # DAX Tooltip
    tempo_global = blo_grp['Tempo da Página'].mean()
    blo_grp['Diff Retencao'] = np.where(tempo_global==0, 0, (blo_grp['Tempo da Página'] - tempo_global) / tempo_global)
    blo_grp['Texto_Dir'] = np.where(blo_grp['Diff Retencao'] > 0, "acima", "abaixo")
    blo_grp['Tooltip'] = "⏳ Retenção: " + (blo_grp['Diff Retencao'].abs()*100).map('{:.1f}%'.format) + " " + blo_grp['Texto_Dir'] + " da média.<br>🎯 Conversão: Gerou " + blo_grp['Clicks'].astype(str) + " cliques."

    # ==========================================
    # 🧠 LINKEDIN (Agrupado por Link/Postagem)
    # ==========================================
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tamanho'] = np.where(lin['Texto'].str.len() < 500, "Curto (Teaser)", np.where(lin['Texto'].str.len() < 1200, "Médio (Padrão)", "Longo"))
    lin['Tipo'] = np.where(lin['Texto'].str.contains('parceria|solução', case=False, na=False), "Comercial", "Educativo")
    
    # AGRUPANDO POSTS IGUAIS
    lin_grp = lin.groupby(['Link da Postagem', 'Texto', 'Data da postagem', 'Tag Produto', 'Tamanho', 'Tipo']).agg({
        'Curtidas': 'sum',
        'Comentários': 'sum',
        'Shares': 'sum',
        'Seguidores': 'max' # Mantém o maior número de seguidores do dia
    }).reset_index()

    lin_grp['Total Engajamento'] = lin_grp['Curtidas'] + lin_grp['Comentários'] + lin_grp['Shares']
    lin_grp['Taxa Engajamento'] = np.where(lin_grp['Seguidores']==0, 0, lin_grp['Total Engajamento'] / lin_grp['Seguidores'])

    # ==========================================
    # 🌐 OVERVIEW FULL BLAST
    # ==========================================
    r_lin = lin_grp.groupby(['Data da postagem', 'Tag Produto'])['Total Engajamento'].sum().reset_index().rename(columns={'Data da postagem': 'Data'})
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index()
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas'].sum().reset_index().rename(columns={'Data de Envio': 'Data'})

    over = pd.merge(r_lin, r_blo, on=['Data', 'Tag Produto'], how='outer')
    over = pd.merge(over, r_mai, on=['Data', 'Tag Produto'], how='outer').fillna(0)
    over['Impacto Total'] = over['Total Engajamento'] + over['Views'] + over['Aberturas']
    over = over.dropna(subset=['Data']).sort_values('Data')

    return over, lin_grp, blo_grp, mai_grp

df_overview, df_linkedin, df_blog, df_mailchimp = processar_dados_dax()

# ==========================================
# INTERFACE DE NAVEGAÇÃO
# ==========================================
st.sidebar.title("🎯 Command Center")
menu = st.sidebar.radio("Visões:", ["🌐 Overview (Full Blast)", "💼 LinkedIn Lab", "📧 Mailchimp", "📝 Blog (SEO)"])

if menu == "🌐 Overview (Full Blast)":
    st.title("Evolução Full Blast")
    evo = df_overview.groupby('Data').sum(numeric_only=True).reset_index()
    
    fig = px.area(evo, x='Data', y='Impacto Total', color_discrete_sequence=[CORES['Total']], custom_data=['Total Engajamento', 'Views', 'Aberturas'])
    fig.update_traces(hovertemplate="<b>Data:</b> %{x}<br><b>Impacto Total:</b> %{y}<br><br>LinkedIn: %{customdata[0]}<br>Blog: %{customdata[1]}<br>Mailchimp: %{customdata[2]}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Tração por Produto")
    rank = df_overview.groupby('Tag Produto')['Impacto Total'].sum().reset_index().sort_values('Impacto Total')
    st.plotly_chart(px.bar(rank[rank['Tag Produto'] != 'Institucional / Outros'], y='Tag Produto', x='Impacto Total', orientation='h', color_discrete_sequence=[CORES['LinkedIn']]), use_container_width=True)

elif menu == "💼 LinkedIn Lab":
    st.title("LinkedIn: Desempenho de IA & Engajamento")
    ab = df_linkedin.groupby(['Tipo', 'Tamanho'])['Total Engajamento'].mean().reset_index()
    fig = px.bar(ab, x='Tipo', y='Total Engajamento', color='Tamanho', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detalhe das Postagens")
    st.dataframe(
        df_linkedin[['Data da postagem', 'Tag Produto', 'Tipo', 'Tamanho', 'Total Engajamento', 'Taxa Engajamento', 'Link da Postagem']].sort_values('Total Engajamento', ascending=False),
        column_config={
            "Taxa Engajamento": st.column_config.NumberColumn(format="%.2f%%"),
            "Link da Postagem": st.column_config.LinkColumn("Acessar Post")
        },
        use_container_width=True
    )

elif menu == "📧 Mailchimp":
    st.title("Mailchimp: Abertura vs CTOR")
    fig = px.scatter(
        df_mailchimp, x='Taxa de Abertura', y='CTOR', size='Qtd Enviados', color='Tag Produto', hover_name='Título',
        custom_data=['Tooltip', 'Ignorados', 'Aberturas']
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br><br>Taxa de Abertura: %{x:.1%}<br>CTOR: %{y:.1%}<br><br>📊 Aberturas: %{customdata[2]}<br>❌ Ignorados: %{customdata[1]}<br><br><i>%{customdata[0]}</i><extra></extra>")
    fig.add_vline(x=df_mailchimp['Taxa de Abertura'].mean(), line_dash="dot", line_color="red")
    fig.add_hline(y=df_mailchimp['CTOR'].mean(), line_dash="dot", line_color="red")
    fig.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consolidado de Campanhas (Agrupadas)")
    st.dataframe(
        df_mailchimp[['Data de Envio', 'Título', 'Tag Produto', 'Qtd Enviados', 'Taxa de Abertura', 'CTOR']].sort_values('CTOR', ascending=False),
        column_config={
            "Taxa de Abertura": st.column_config.NumberColumn(format="%.1f%%"),
            "CTOR": st.column_config.NumberColumn(format="%.1f%%")
        },
        use_container_width=True
    )

elif menu == "📝 Blog":
    st.title("Blog: Retenção e Conversão SEO")
    fig = px.scatter(
        df_blog, x='Views', y='Tempo da Página', size='Clicks', color='Tag Produto', hover_name='Título Post',
        custom_data=['Tooltip', 'Taxa Conversão']
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Taxa Conversão: %{customdata[1]:.1%}<br>Views: %{x}<br>Tempo: %{y}s<br><br>%{customdata[0]}<extra></extra>")
    fig.add_hline(y=df_blog['Tempo da Página'].mean(), line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Auditoria de Artigos")
    st.dataframe(
        df_blog[['Data', 'Título Post', 'Tag Produto', 'Views', 'Tempo da Página', 'Taxa Conversão', 'URL']].sort_values('Views', ascending=False),
        column_config={
            "Taxa Conversão": st.column_config.NumberColumn(format="%.1f%%"),
            "URL": st.column_config.LinkColumn("Ler Artigo")
        },
        use_container_width=True
    )
