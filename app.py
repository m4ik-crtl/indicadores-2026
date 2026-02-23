import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AIQON | Command Center", layout="wide", initial_sidebar_state="expanded")
CORES = {'LinkedIn': '#0A66C2', 'Mailchimp': '#F97316', 'Blog': '#10B981', 'Total': '#1E293B', 'Fundo': '#F8FAFC'}

@st.cache_data
def carregar_dados():
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')

    # Convertendo datas
    lin['Data'] = pd.to_datetime(lin['Data da postagem']).dt.date
    blo['Data'] = pd.to_datetime(blo['Data']).dt.date
    mai['Data de Envio'] = pd.to_datetime(mai['Data de Envio']).dt.date

    # --- TAG DE PRODUTO ---
    def tag_produto(t):
        t = str(t).lower()
        if 'action1' in t: return 'Action1'
        elif 'netwrix' in t: return 'Netwrix'
        elif '42crunch' in t: return '42Crunch'
        elif 'syxsense' in t: return 'Syxsense'
        elif 'ox ' in t or 'ox security' in t: return 'Ox Security'
        elif 'cynet' in t: return 'Cynet'
        elif 'easy inventory' in t: return 'Easy Inventory'
        elif 'keepit' in t: return 'Keepit'
        elif 'grip' in t: return 'Grip'
        elif 'manage engine' in t: return 'Manage Engine'
        elif 'wallarm' in t: return 'Wallarm'
        else: return 'Institucional / Outros'

    # ==========================================
    # 🧠 MAILCHIMP (AGRUPANDO POR TÍTULO)
    # ==========================================
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    mai['Aberturas_Abs'] = np.round(mai['Qtd Enviados'] * mai['Taxa de Abertura']).astype(int)
    mai['Cliques_Abs'] = np.round(mai['Qtd Enviados'] * mai['Clicks']).astype(int)

    mai_grp = mai.groupby(['Título', 'Tag Produto']).agg({
        'Qtd Enviados': 'sum', 'Aberturas_Abs': 'sum', 'Cliques_Abs': 'sum', 'Data de Envio': 'max'
    }).reset_index()

    # Recalculando DAX corretamente a partir dos absolutos
    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados']==0, 0, mai_grp['Aberturas_Abs'] / mai_grp['Qtd Enviados'])
    mai_grp['Taxa de Clique por Abertura (CTOR)'] = np.where(mai_grp['Aberturas_Abs']==0, 0, mai_grp['Cliques_Abs'] / mai_grp['Aberturas_Abs'])
    mai_grp['Total Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas_Abs']
    mai_grp['Plataforma'] = "Mailchimp"
    mai_grp['Link'] = "Não aplicável"

    # ==========================================
    # 🧠 BLOG (AGRUPANDO POR URL)
    # ==========================================
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Título Post'] = blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.upper()

    blo_grp = blo.groupby(['URL', 'Título Post', 'Tag Produto']).agg({
        'Views': 'sum', 'Clicks': 'sum', 'Tempo da Página': 'mean', 'Data': 'max'
    }).reset_index()

    blo_grp['Taxa Conversão Blog'] = np.where(blo_grp['Views']==0, 0, blo_grp['Clicks'] / blo_grp['Views'])
    blo_grp['Plataforma'] = "Blog"
    blo_grp['Link'] = blo_grp['URL']

    # ==========================================
    # 🧠 LINKEDIN (AGRUPANDO POR POSTAGEM)
    # ==========================================
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tamanho do Post'] = np.where(lin['Texto'].str.len() < 500, "Curto (Teaser)", np.where(lin['Texto'].str.len() < 1200, "Médio (Padrão)", "Longo (Artigo)"))
    lin['Tipo Conteúdo'] = np.where(lin['Texto'].str.contains('parceria|solução', case=False, na=False), "Comercial/Venda", "Editorial/Educativo")

    lin_grp = lin.groupby(['Link da Postagem', 'Título', 'Tag Produto', 'Tamanho do Post', 'Tipo Conteúdo']).agg({
        'Curtidas': 'sum', 'Comentários': 'sum', 'Shares': 'sum', 'Seguidores': 'max', 'Data': 'max'
    }).reset_index()

    lin_grp['Total Engajamento'] = lin_grp['Curtidas'] + lin_grp['Comentários'] + lin_grp['Shares']
    lin_grp['Taxa Engajamento (ER)'] = np.where(lin_grp['Seguidores']==0, 0, lin_grp['Total Engajamento'] / lin_grp['Seguidores'])
    lin_grp['Plataforma'] = "LinkedIn"
    lin_grp['Link'] = lin_grp['Link da Postagem']

    # ==========================================
    # 🌐 FULL BLAST (OVERVIEW)
    # ==========================================
    r_lin = lin_grp.groupby(['Data', 'Tag Produto'])['Total Engajamento'].sum().reset_index()
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index()
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas_Abs'].sum().reset_index().rename(columns={'Data de Envio': 'Data'})

    over = pd.merge(r_lin, r_blo, on=['Data', 'Tag Produto'], how='outer')
    over = pd.merge(over, r_mai, on=['Data', 'Tag Produto'], how='outer').fillna(0)
    over['Impacto Total'] = over['Total Engajamento'] + over['Views'] + over['Aberturas_Abs']
    over = over.dropna(subset=['Data']).sort_values('Data')

    # CRIANDO A LISTA DE DADOS UNIFICADA
    df_lista = pd.concat([
        lin_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Total Engajamento']].rename(columns={'Total Engajamento': 'Tração (Absoluto)'}),
        blo_grp[['Data', 'Título Post', 'Plataforma', 'Link', 'Tag Produto', 'Views']].rename(columns={'Título Post': 'Título', 'Views': 'Tração (Absoluto)'}),
        mai_grp[['Data de Envio', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Aberturas_Abs']].rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Tração (Absoluto)'})
    ]).sort_values('Data', ascending=False)

    return over, lin_grp, blo_grp, mai_grp, df_lista

df_over, df_lin, df_blo, df_mai, df_lista = carregar_dados()

# ==========================================
# SIDEBAR DE FILTROS (ALLSELECTED)
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
st.sidebar.title("Comandos")
menu = st.sidebar.radio("Navegação:", ["🌐 Overview (Full Blast)", "💼 LinkedIn Lab", "📧 Mailchimp", "📝 Blog (SEO)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Globais")

min_d = min(df_over['Data'].min(), df_lin['Data'].min(), df_blo['Data'].min(), df_mai['Data de Envio'].min())
max_d = max(df_over['Data'].max(), df_lin['Data'].max(), df_blo['Data'].max(), df_mai['Data de Envio'].max())

datas = st.sidebar.date_input("Filtrar Período", [min_d, max_d], min_value=min_d, max_value=max_d)
prod_disp = df_over['Tag Produto'].unique().tolist()
filtro_prod = st.sidebar.multiselect("Produto Promovido", prod_disp, default=prod_disp)

if len(datas) == 2:
    over_filt = df_over[(df_over['Data'] >= datas[0]) & (df_over['Data'] <= datas[1]) & (df_over['Tag Produto'].isin(filtro_prod))]
    lin_filt = df_lin[(df_lin['Data'] >= datas[0]) & (df_lin['Data'] <= datas[1]) & (df_lin['Tag Produto'].isin(filtro_prod))]
    blo_filt = df_blo[(df_blo['Data'] >= datas[0]) & (df_blo['Data'] <= datas[1]) & (df_blo['Tag Produto'].isin(filtro_prod))]
    mai_filt = df_mai[(df_mai['Data de Envio'] >= datas[0]) & (df_mai['Data de Envio'] <= datas[1]) & (df_mai['Tag Produto'].isin(filtro_prod))]
    lista_filt = df_lista[(df_lista['Data'] >= datas[0]) & (df_lista['Data'] <= datas[1]) & (df_lista['Tag Produto'].isin(filtro_prod))]
else:
    over_filt, lin_filt, blo_filt, mai_filt, lista_filt = df_over, df_lin, df_blo, df_mai, df_lista

# ==========================================
# MÓDULOS (TELAS)
# ==========================================

if menu == "🌐 Overview (Full Blast)":
    st.title("Visão Geral: Impacto da Marca (Full Blast)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impacto Total (Toques)", f"{int(over_filt['Impacto Total'].sum()):,}")
    c2.metric("Total Social (Engajamento)", f"{int(over_filt['Total Engajamento'].sum()):,}")
    c3.metric("Total Views Blog", f"{int(over_filt['Views'].sum()):,}")
    c4.metric("Total Aberturas Email", f"{int(over_filt['Aberturas_Abs'].sum()):,}")

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Evolução Diária do Barulho")
        evo = over_filt.groupby('Data').sum(numeric_only=True).reset_index()
        fig = px.area(evo, x='Data', y='Impacto Total', color_discrete_sequence=[CORES['Total']], custom_data=['Total Engajamento', 'Views', 'Aberturas_Abs'])
        fig.update_traces(hovertemplate="<b>Data:</b> %{x}<br><b>Impacto Total:</b> %{y:,.0f}<br><br>LinkedIn: %{customdata[0]:,.0f}<br>Blog: %{customdata[1]:,.0f}<br>Mailchimp: %{customdata[2]:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tração por Produto")
        rank = over_filt.groupby('Tag Produto')['Impacto Total'].sum().reset_index().sort_values('Impacto Total')
        st.plotly_chart(px.bar(rank[rank['Tag Produto'] != 'Institucional / Outros'], y='Tag Produto', x='Impacto Total', orientation='h', color_discrete_sequence=[CORES['LinkedIn']]), use_container_width=True)

    st.subheader("Lista Mestra de Ativos de Marketing")
    st.dataframe(
        lista_filt[['Data', 'Plataforma', 'Título', 'Tag Produto', 'Tração (Absoluto)', 'Link']],
        column_config={"Link": st.column_config.LinkColumn("🔗 Acessar Original"), "Tração (Absoluto)": st.column_config.NumberColumn(format="%d")},
        use_container_width=True, hide_index=True
    )

elif menu == "💼 LinkedIn Lab":
    st.title("LinkedIn: Desempenho e IA")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média Engajamento/Post", f"{lin_filt['Total Engajamento'].mean():.1f}" if not lin_filt.empty else "0")
    c2.metric("Taxa Média de Engajamento (ER)", f"{(lin_filt['Total Engajamento'].sum() / lin_filt['Seguidores'].sum() if lin_filt['Seguidores'].sum() > 0 else 0):.2%}")
    c3.metric("Total de Publicações", f"{len(lin_filt)}")

    st.markdown("---")
    ab = lin_filt.groupby(['Tipo Conteúdo', 'Tamanho do Post'])['Total Engajamento'].mean().reset_index()
    st.plotly_chart(px.bar(ab, x='Tipo Conteúdo', y='Total Engajamento', color='Tamanho do Post', barmode='group'), use_container_width=True)

    st.dataframe(
        lin_filt[['Data', 'Título', 'Tag Produto', 'Tipo Conteúdo', 'Tamanho do Post', 'Total Engajamento', 'Taxa Engajamento (ER)', 'Link da Postagem']].sort_values('Total Engajamento', ascending=False),
        column_config={"Taxa Engajamento (ER)": st.column_config.NumberColumn(format="%.2f%%"), "Link da Postagem": st.column_config.LinkColumn("Acessar")},
        use_container_width=True, hide_index=True
    )

elif menu == "📧 Mailchimp":
    st.title("Mailchimp: Conversão de E-mail")
    # Lógica DAX Global
    ctor_global = mai_filt['Cliques_Abs'].sum() / mai_filt['Aberturas_Abs'].sum() if mai_filt['Aberturas_Abs'].sum() > 0 else 0
    taxa_abertura_global = mai_filt['Aberturas_Abs'].sum() / mai_filt['Qtd Enviados'].sum() if mai_filt['Qtd Enviados'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Taxa de Abertura Global", f"{taxa_abertura_global:.2%}")
    c2.metric("Taxa CTOR Global", f"{ctor_global:.2%}")
    c3.metric("E-mails Disparados", f"{mai_filt['Qtd Enviados'].sum():,}")

    st.markdown("---")
    mai_plot = mai_filt.copy()
    mai_plot['Diff CTOR'] = np.where(ctor_global==0, 0, (mai_plot['Taxa de Clique por Abertura (CTOR)'] - ctor_global) / ctor_global)
    mai_plot['Texto Performance Tooltip'] = "Este e-mail performou " + (mai_plot['Diff CTOR']*100).map('{:+.1f}%'.format) + " vs média global."

    fig = px.scatter(
        mai_plot, x='Taxa de Abertura', y='Taxa de Clique por Abertura (CTOR)', size='Qtd Enviados', color='Tag Produto', hover_name='Título',
        custom_data=['Texto Performance Tooltip', 'Total Ignorados', 'Aberturas_Abs']
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br><br>Abertura: %{x:.1%}<br>CTOR: %{y:.1%}<br>📊 Aberturas: %{customdata[2]:,.0f}<br>❌ Ignorados: %{customdata[1]:,.0f}<br><br><i>%{customdata[0]}</i><extra></extra>")
    fig.add_vline(x=taxa_abertura_global, line_dash="dot", line_color="red")
    fig.add_hline(y=ctor_global, line_dash="dot", line_color="red")
    fig.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        mai_filt[['Data de Envio', 'Título', 'Tag Produto', 'Qtd Enviados', 'Taxa de Abertura', 'Taxa de Clique por Abertura (CTOR)']].sort_values('Taxa de Clique por Abertura (CTOR)', ascending=False),
        column_config={"Taxa de Abertura": st.column_config.NumberColumn(format="%.2f%%"), "Taxa de Clique por Abertura (CTOR)": st.column_config.NumberColumn(format="%.2f%%")},
        use_container_width=True, hide_index=True
    )

elif menu == "📝 Blog":
    st.title("Blog: Retenção e SEO")
    tempo_global = blo_filt['Tempo da Página'].mean() if not blo_filt.empty else 0
    tx_conversao_global = blo_filt['Clicks'].sum() / blo_filt['Views'].sum() if blo_filt['Views'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Tempo Médio Global (s)", f"{tempo_global:.1f}s")
    c2.metric("Taxa Conversão Global", f"{tx_conversao_global:.2%}")
    c3.metric("Leituras (Views)", f"{blo_filt['Views'].sum():,}")

    st.markdown("---")
    blo_plot = blo_filt.copy()
    blo_plot['Diff Retencao'] = np.where(tempo_global==0, 0, (blo_plot['Tempo da Página'] - tempo_global) / tempo_global)
    blo_plot['Texto_Dir'] = np.where(blo_plot['Diff Retencao'] > 0, "acima", "abaixo")
    blo_plot['Texto Performance Blog'] = "⏳ Retenção: " + (blo_plot['Diff Retencao'].abs()*100).map('{:.1f}%'.format) + " " + blo_plot['Texto_Dir'] + " da média do blog.<br>🎯 Conversão: Gerou um total de " + blo_plot['Clicks'].astype(str) + " cliques."

    fig = px.scatter(
        blo_plot, x='Views', y='Tempo da Página', size='Clicks', color='Tag Produto', hover_name='Título Post', custom_data=['Texto Performance Blog', 'Taxa Conversão Blog']
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Taxa Conversão: %{customdata[1]:.1%}<br>Views: %{x}<br>Tempo: %{y:.0f}s<br><br>%{customdata[0]}<extra></extra>")
    fig.add_hline(y=tempo_global, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        blo_filt[['Data', 'Título Post', 'Tag Produto', 'Views', 'Tempo da Página', 'Taxa Conversão Blog', 'URL']].sort_values('Views', ascending=False),
        column_config={"Taxa Conversão Blog": st.column_config.NumberColumn(format="%.2f%%"), "URL": st.column_config.LinkColumn("Ler Artigo")},
        use_container_width=True, hide_index=True
    )
