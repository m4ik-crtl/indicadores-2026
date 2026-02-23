import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AIQON | Command Center", layout="wide", initial_sidebar_state="expanded")

# --- Identidade Visual B2B ---
CORES_PRODUTOS = {
    'Action1': '#0A66C2',           
    'Netwrix': '#DC2626',           
    '42Crunch': '#F97316',          
    'Ox Security': '#8B5CF6',       
    'Cynet': '#06B6D4',             
    'Easy Inventory': '#F59E0B',    
    'Keepit': '#10B981',            
    'Grip': '#14B8A6',              
    'Manage Engine': '#3B82F6',     
    'Wallarm': '#E11D48',           
    'Institucional / Outros': '#94A3B8' 
}

# Formatador PT-BR
def f_br(num, is_pct=False):
    if pd.isna(num): return "0"
    if is_pct: return f"{num * 100:.2f}%".replace('.', ',')
    return f"{int(num):,}".replace(',', '.')

@st.cache_data
def carregar_dados():
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')

    lin['Data'] = pd.to_datetime(lin['Data da postagem']).dt.date
    blo['Data'] = pd.to_datetime(blo['Data']).dt.date
    mai['Data de Envio'] = pd.to_datetime(mai['Data de Envio']).dt.date

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

    # --- MAILCHIMP ---
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    # Matemática: Calculando os absolutos reais a partir da taxa
    mai['Aberturas_Abs'] = (mai['Qtd Enviados'] * mai['Taxa de Abertura']).round().astype(int)
    mai['Cliques_Abs'] = (mai['Qtd Enviados'] * mai['Clicks']).round().astype(int)

    mai_grp = mai.groupby(['Título', 'Tag Produto']).agg({
        'Qtd Enviados': 'sum', 'Aberturas_Abs': 'sum', 'Cliques_Abs': 'sum', 'Data de Envio': 'max'
    }).reset_index()

    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados']==0, 0, mai_grp['Aberturas_Abs'] / mai_grp['Qtd Enviados'])
    mai_grp['CTOR'] = np.where(mai_grp['Aberturas_Abs']==0, 0, mai_grp['Cliques_Abs'] / mai_grp['Aberturas_Abs'])
    mai_grp['Total Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas_Abs']
    mai_grp['Plataforma'] = "Mailchimp"
    mai_grp['Link'] = "N/A"

    # --- BLOG ---
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Título Post'] = blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.upper()

    blo_grp = blo.groupby(['URL', 'Título Post', 'Tag Produto']).agg({
        'Views': 'sum', 'Clicks': 'sum', 'Tempo da Página': 'mean', 'Data': 'max'
    }).reset_index()

    blo_grp['Taxa Conversão Blog'] = np.where(blo_grp['Views']==0, 0, blo_grp['Clicks'] / blo_grp['Views'])
    blo_grp['Plataforma'] = "Blog"
    blo_grp['Link'] = blo_grp['URL']

    # --- LINKEDIN ---
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tamanho do Post'] = np.where(lin['Texto'].str.len() < 500, "Curto", np.where(lin['Texto'].str.len() < 1200, "Médio", "Longo"))
    lin['Tipo Conteúdo'] = np.where(lin['Texto'].str.contains('parceria|solução', case=False, na=False), "Comercial/Venda", "Editorial/Educativo")

    lin_grp = lin.groupby(['Link da Postagem', 'Título', 'Tag Produto', 'Tamanho do Post', 'Tipo Conteúdo']).agg({
        'Curtidas': 'sum', 'Comentários': 'sum', 'Shares': 'sum', 'Seguidores': 'max', 'Data': 'max'
    }).reset_index()

    lin_grp['Total Engajamento'] = lin_grp['Curtidas'] + lin_grp['Comentários'] + lin_grp['Shares']
    lin_grp['Taxa Engajamento (ER)'] = np.where(lin_grp['Seguidores']==0, 0, lin_grp['Total Engajamento'] / lin_grp['Seguidores'])
    lin_grp['Plataforma'] = "LinkedIn"
    lin_grp['Link'] = lin_grp['Link da Postagem']

    # --- OVERVIEW ---
    r_lin = lin_grp.groupby(['Data', 'Tag Produto'])['Total Engajamento'].sum().reset_index()
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index()
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas_Abs'].sum().reset_index().rename(columns={'Data de Envio': 'Data'})

    over = pd.merge(r_lin, r_blo, on=['Data', 'Tag Produto'], how='outer')
    over = pd.merge(over, r_mai, on=['Data', 'Tag Produto'], how='outer').fillna(0)
    over['Impacto Total'] = (over['Total Engajamento'] + over['Views'] + over['Aberturas_Abs']).astype(int)
    over = over.dropna(subset=['Data']).sort_values('Data')

    df_lista = pd.concat([
        lin_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Total Engajamento']].rename(columns={'Total Engajamento': 'Tração (Absoluto)'}),
        blo_grp[['Data', 'Título Post', 'Plataforma', 'Link', 'Tag Produto', 'Views']].rename(columns={'Título Post': 'Título', 'Views': 'Tração (Absoluto)'}),
        mai_grp[['Data de Envio', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Aberturas_Abs']].rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Tração (Absoluto)'})
    ]).sort_values('Data', ascending=False)

    return over, lin_grp, blo_grp, mai_grp, df_lista

df_over, df_lin, df_blo, df_mai, df_lista = carregar_dados()

# ==========================================
# SIDEBAR / FILTROS
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
st.sidebar.title("Comandos Analíticos")
menu = st.sidebar.radio("Navegação:", ["🌐 Visão Geral", "💼 LinkedIn", "📧 Mailchimp", "📝 Blog (SEO)"])
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
# TELAS DO DASHBOARD
# ==========================================

if menu == "🌐 Visão Geral":
    st.title("Visão Geral: Impacto da Marca (Full Blast)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impacto Total (Toques)", f_br(over_filt['Impacto Total'].sum()))
    c2.metric("Total Social (LinkedIn)", f_br(over_filt['Total Engajamento'].sum()))
    c3.metric("Total Views Blog", f_br(over_filt['Views'].sum()))
    c4.metric("Total Aberturas Email", f_br(over_filt['Aberturas_Abs'].sum()))

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Evolução Diária do Barulho")
        evo = over_filt.groupby('Data').sum(numeric_only=True).reset_index()
        fig = px.area(evo, x='Data', y='Impacto Total', color_discrete_sequence=['#1E293B'], custom_data=['Total Engajamento', 'Views', 'Aberturas_Abs'])
        fig.update_traces(hovertemplate="<b>Data:</b> %{x}<br><b>Impacto Total:</b> %{y}<br><br>LinkedIn: %{customdata[0]}<br>Blog: %{customdata[1]}<br>Mailchimp: %{customdata[2]}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tração por Produto")
        rank = over_filt.groupby('Tag Produto')['Impacto Total'].sum().reset_index().sort_values('Impacto Total')
        fig2 = px.bar(rank[rank['Tag Produto'] != 'Institucional / Outros'], y='Tag Produto', x='Impacto Total', orientation='h', color='Tag Produto', color_discrete_map=CORES_PRODUTOS)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Lista Mestra de Ativos de Marketing")
    st.dataframe(
        lista_filt[['Data', 'Plataforma', 'Título', 'Tag Produto', 'Tração (Absoluto)', 'Link']],
        column_config={"Link": st.column_config.LinkColumn("🔗 Acessar Original"), "Tração (Absoluto)": st.column_config.NumberColumn(format="%d")},
        use_container_width=True, hide_index=True
    )

elif menu == "💼 LinkedIn":
    st.title("LinkedIn: Desempenho e IA")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média Engajamento/Post", f_br(lin_filt['Total Engajamento'].mean()))
    c2.metric("Taxa Média de Engajamento (ER)", f_br((lin_filt['Total Engajamento'].sum() / lin_filt['Seguidores'].sum() if lin_filt['Seguidores'].sum() > 0 else 0), True))
    c3.metric("Total de Publicações", f_br(len(lin_filt)))

    st.markdown("---")
    ab = lin_filt.groupby(['Tipo Conteúdo', 'Tamanho do Post'])['Total Engajamento'].mean().reset_index()
    st.plotly_chart(px.bar(ab, x='Tipo Conteúdo', y='Total Engajamento', color='Tamanho do Post', barmode='group'), use_container_width=True)

    st.dataframe(
        lin_filt[['Data', 'Título', 'Tag Produto', 'Tipo Conteúdo', 'Tamanho do Post', 'Total Engajamento', 'Taxa Engajamento (ER)', 'Link da Postagem']].sort_values('Total Engajamento', ascending=False),
        column_config={"Taxa Engajamento (ER)": st.column_config.NumberColumn(format="%.2f%%"), "Link da Postagem": st.column_config.LinkColumn("🔗 Acessar")},
        use_container_width=True, hide_index=True
    )

elif menu == "📧 Mailchimp":
    st.title("Mailchimp: Conversão e Funil")
    ctor_global = mai_filt['Cliques_Abs'].sum() / mai_filt['Aberturas_Abs'].sum() if mai_filt['Aberturas_Abs'].sum() > 0 else 0
    taxa_abertura_global = mai_filt['Aberturas_Abs'].sum() / mai_filt['Qtd Enviados'].sum() if mai_filt['Qtd Enviados'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Taxa de Abertura Global", f_br(taxa_abertura_global, True))
    c2.metric("Taxa CTOR Global", f_br(ctor_global, True))
    c3.metric("E-mails Disparados", f_br(mai_filt['Qtd Enviados'].sum()))

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Funil de Retenção")
        fig_funil = go.Figure(go.Funnel(
            y=['1. E-mails Enviados', '2. Aberturas (Interesse)', '3. Cliques (Intenção)'],
            x=[mai_filt['Qtd Enviados'].sum(), mai_filt['Aberturas_Abs'].sum(), mai_filt['Cliques_Abs'].sum()],
            textinfo="value+percent initial",
            marker={"color": ["#1E293B", "#F97316", "#10B981"]}
        ))
        fig_funil.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_funil, use_container_width=True)

    with col2:
        st.subheader("Abertura vs CTOR")
        mai_plot = mai_filt.copy()
        mai_plot['Diff CTOR'] = np.where(ctor_global==0, 0, (mai_plot['CTOR'] - ctor_global) / ctor_global)
        mai_plot['Tooltip'] = "E-mail performou " + (mai_plot['Diff CTOR']*100).map('{:+.2f}%'.format).str.replace('.', ',') + " vs média global."

        fig_scatter = px.scatter(
            mai_plot, x='Taxa de Abertura', y='CTOR', size='Qtd Enviados', color='Tag Produto', hover_name='Título',
            color_discrete_map=CORES_PRODUTOS, custom_data=['Tooltip', 'Total Ignorados', 'Aberturas_Abs']
        )
        fig_scatter.update_traces(hovertemplate="<b>%{hovertext}</b><br>Abertura: %{x:.2%}<br>CTOR: %{y:.2%}<br>📊 Aberturas: %{customdata[2]}<br>❌ Ignorados: %{customdata[1]}<br><br><i>%{customdata[0]}</i><extra></extra>")
        fig_scatter.add_vline(x=taxa_abertura_global, line_dash="dot", line_color="red")
        fig_scatter.add_hline(y=ctor_global, line_dash="dot", line_color="red")
        fig_scatter.update_layout(xaxis_tickformat='.2%', yaxis_tickformat='.2%', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Auditoria de Campanhas")
    st.dataframe(
        mai_filt[['Data de Envio', 'Título', 'Tag Produto', 'Qtd Enviados', 'Aberturas_Abs', 'Cliques_Abs', 'Taxa de Abertura', 'CTOR']].sort_values('CTOR', ascending=False),
        column_config={"Taxa de Abertura": st.column_config.NumberColumn(format="%.2f%%"), "CTOR": st.column_config.NumberColumn(format="%.2f%%")},
        use_container_width=True, hide_index=True
    )

elif menu == "📝 Blog (SEO)":
    st.title("Blog: Retenção e SEO")
    tempo_global = blo_filt['Tempo da Página'].mean() if not blo_filt.empty else 0
    tx_conversao_global = blo_filt['Clicks'].sum() / blo_filt['Views'].sum() if blo_filt['Views'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Tempo Médio Global", f_br(tempo_global) + "s")
    c2.metric("Taxa Conversão Global", f_br(tx_conversao_global, True))
    c3.metric("Total de Leituras", f_br(blo_filt['Views'].sum()))

    st.markdown("---")
    blo_plot = blo_filt.copy()
    blo_plot['Diff Retencao'] = np.where(tempo_global==0, 0, (blo_plot['Tempo da Página'] - tempo_global) / tempo_global)
    blo_plot['Texto_Dir'] = np.where(blo_plot['Diff Retencao'] > 0, "acima", "abaixo")
    blo_plot['Tooltip'] = "⏳ Retenção: " + (blo_plot['Diff Retencao'].abs()*100).map('{:.1f}%'.format).str.replace('.', ',') + " " + blo_plot['Texto_Dir'] + " da média.<br>🎯 Conversão: Gerou " + blo_plot['Clicks'].astype(str) + " cliques."

    fig = px.scatter(
        blo_plot, x='Views', y='Tempo da Página', size='Clicks', color='Tag Produto', hover_name='Título Post', 
        color_discrete_map=CORES_PRODUTOS, custom_data=['Tooltip', 'Taxa Conversão Blog']
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Taxa Conversão: %{customdata[1]:.2%}<br>Views: %{x}<br>Tempo: %{y:.0f}s<br><br>%{customdata[0]}<extra></extra>")
    fig.add_hline(y=tempo_global, line_dash="dot", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        blo_filt[['Data', 'Título Post', 'Tag Produto', 'Views', 'Tempo da Página', 'Taxa Conversão Blog', 'URL']].sort_values('Views', ascending=False),
        column_config={"Taxa Conversão Blog": st.column_config.NumberColumn(format="%.2f%%"), "URL": st.column_config.LinkColumn("🔗 Ler Artigo")},
        use_container_width=True, hide_index=True
    )
