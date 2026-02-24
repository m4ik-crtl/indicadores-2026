import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats

st.set_page_config(page_title="AIQON | IA Command Center", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🎨 IDENTIDADE VISUAL B2B
# ==========================================
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
    'Institucional / Outros': '#475569'
}

# 🇧🇷 FORMATADOR BRASILEIRO ESTRITO
def f_br(num, is_pct=False):
    if pd.isna(num): return "0"
    if is_pct:
        return f"{num * 100:.2f}%".replace('.', ',')
    else:
        return f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# ==========================================
# 🧠 MOTOR DE DADOS & DAX
# ==========================================
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
    mai['Aberturas_Abs'] = (mai['Qtd Enviados'] * mai['Taxa de Abertura']).fillna(0).round().astype(int)
    mai['Cliques_Abs'] = (mai['Qtd Enviados'] * mai['Clicks']).fillna(0).round().astype(int)

    mai_grp = mai.groupby(['Título', 'Tag Produto']).agg({
        'Qtd Enviados': 'sum', 'Aberturas_Abs': 'sum', 'Cliques_Abs': 'sum', 'Data de Envio': 'max'
    }).reset_index()

    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados']==0, 0, mai_grp['Aberturas_Abs'] / mai_grp['Qtd Enviados'])
    mai_grp['CTOR'] = np.where(mai_grp['Aberturas_Abs']==0, 0, mai_grp['Cliques_Abs'] / mai_grp['Aberturas_Abs'])
    mai_grp['Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas_Abs']
    mai_grp['Plataforma'], mai_grp['Link'] = "Mailchimp", "N/A"

    # --- BLOG ---
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Título'] = blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.title()

    blo_grp = blo.groupby(['URL', 'Título', 'Tag Produto']).agg({
        'Views': 'sum', 'Clicks': 'sum', 'Tempo da Página': 'mean', 'Data': 'max'
    }).reset_index()

    blo_grp['Taxa Conversão'] = np.where(blo_grp['Views']==0, 0, blo_grp['Clicks'] / blo_grp['Views'])
    blo_grp['Plataforma'], blo_grp['Link'] = "Blog", blo_grp['URL']

    # --- LINKEDIN ---
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tamanho'] = np.where(lin['Texto'].str.len() < 500, "Curto", np.where(lin['Texto'].str.len() < 1200, "Médio", "Longo"))
    lin['Tipo'] = np.where(lin['Texto'].str.contains('parceria|solução', case=False, na=False), "Comercial/Venda", "Editorial/Educativo")

    lin_grp = lin.groupby(['Link da Postagem', 'Título', 'Tag Produto', 'Tamanho', 'Tipo']).agg({
        'Curtidas': 'sum', 'Comentários': 'sum', 'Shares': 'sum', 'Seguidores': 'max', 'Data': 'max'
    }).reset_index()

    lin_grp['Engajamento'] = lin_grp['Curtidas'] + lin_grp['Comentários'] + lin_grp['Shares']
    lin_grp['Taxa Engajamento (ER)'] = np.where(lin_grp['Seguidores']==0, 0, lin_grp['Engajamento'] / lin_grp['Seguidores'])
    lin_grp['Plataforma'], lin_grp['Link'] = "LinkedIn", lin_grp['Link da Postagem']

    # --- OVERVIEW ---
    r_lin = lin_grp.groupby(['Data', 'Tag Produto'])['Engajamento'].sum().reset_index().rename(columns={'Engajamento': 'Tração'})
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index().rename(columns={'Views': 'Tração'})
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas_Abs'].sum().reset_index().rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Tração'})

    over = pd.concat([r_lin, r_blo, r_mai]).groupby(['Data', 'Tag Produto'])['Tração'].sum().reset_index()
    over = over.sort_values('Data')

    lista = pd.concat([
        lin_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Engajamento']].rename(columns={'Engajamento': 'Cliques/Tração'}),
        blo_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Views']].rename(columns={'Views': 'Cliques/Tração'}),
        mai_grp[['Data de Envio', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Aberturas_Abs']].rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Cliques/Tração'})
    ]).sort_values('Data', ascending=False)

    return over, lin_grp, blo_grp, mai_grp, lista

df_over, df_lin, df_blo, df_mai, df_lista = carregar_dados()

# ==========================================
# SIDEBAR / FILTROS ALLSELECTED
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50)
st.sidebar.title("Comandos")
menu = st.sidebar.radio("Navegação:", ["🌐 Visão Geral", "💼 LinkedIn", "📧 Mailchimp", "📝 Blog", "🤖 IA & Machine Learning"])
st.sidebar.markdown("---")

min_d = min(df_over['Data'].min(), df_lin['Data'].min(), df_blo['Data'].min(), df_mai['Data de Envio'].min())
max_d = max(df_over['Data'].max(), df_lin['Data'].max(), df_blo['Data'].max(), df_mai['Data de Envio'].max())

datas = st.sidebar.date_input("Filtrar Período", [min_d, max_d], min_value=min_d, max_value=max_d)
prod_disp = df_over['Tag Produto'].unique().tolist()
filtro_prod = st.sidebar.multiselect("Produto Promovido", prod_disp, default=prod_disp)

if len(datas) == 2:
    over_f = df_over[(df_over['Data'] >= datas[0]) & (df_over['Data'] <= datas[1]) & (df_over['Tag Produto'].isin(filtro_prod))]
    lin_f = df_lin[(df_lin['Data'] >= datas[0]) & (df_lin['Data'] <= datas[1]) & (df_lin['Tag Produto'].isin(filtro_prod))]
    blo_f = df_blo[(df_blo['Data'] >= datas[0]) & (df_blo['Data'] <= datas[1]) & (df_blo['Tag Produto'].isin(filtro_prod))]
    mai_f = df_mai[(df_mai['Data de Envio'] >= datas[0]) & (df_mai['Data de Envio'] <= datas[1]) & (df_mai['Tag Produto'].isin(filtro_prod))]
    lista_f = df_lista[(df_lista['Data'] >= datas[0]) & (df_lista['Data'] <= datas[1]) & (df_lista['Tag Produto'].isin(filtro_prod))]
else:
    over_f, lin_f, blo_f, mai_f, lista_f = df_over, df_lin, df_blo, df_mai, df_lista

# ==========================================
# MÓDULOS DE RELATÓRIO PADRÃO
# ==========================================

if menu == "🌐 Visão Geral":
    st.title("Visão Geral: Impacto da Marca (Full Blast)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impacto Total (Toques)", f_br(over_f['Tração'].sum()))
    c2.metric("Pico Diário", f_br(over_f.groupby('Data')['Tração'].sum().max() if not over_f.empty else 0))
    c3.metric("Ticket Médio (Ações/Dia)", f_br(over_f.groupby('Data')['Tração'].sum().mean() if not over_f.empty else 0))
    c4.metric("Produtos Promovidos", len(over_f['Tag Produto'].unique()))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Evolução Diária do Barulho")
        evo = over_f.groupby('Data')['Tração'].sum().reset_index()
        fig = px.bar(evo, x='Data', y='Tração', color_discrete_sequence=['#1E293B'])
        if not evo.empty:
            fig.add_hline(y=evo['Tração'].mean(), line_dash="dash", line_color="#DC2626", annotation_text="Média Esperada")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tração por Produto")
        rank = over_f.groupby('Tag Produto')['Tração'].sum().reset_index().sort_values('Tração')
        fig2 = px.bar(rank[rank['Tag Produto'] != 'Institucional / Outros'], y='Tag Produto', x='Tração', orientation='h', color='Tag Produto', color_discrete_map=CORES_PRODUTOS)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Lista Mestra de Ativos")
    st.dataframe(lista_f, column_config={"Link": st.column_config.LinkColumn("🔗 Abrir Origem"), "Cliques/Tração": st.column_config.NumberColumn(format="%d")}, use_container_width=True, hide_index=True)

elif menu == "💼 LinkedIn":
    st.title("LinkedIn: Desempenho Básico")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média Engajamento/Post", f_br(lin_f['Engajamento'].mean()))
    c2.metric("ER Médio", f_br((lin_f['Engajamento'].sum() / lin_f['Seguidores'].sum() if lin_f['Seguidores'].sum() > 0 else 0), True))
    c3.metric("Volume de Posts", f_br(len(lin_f)))

    st.markdown("---")
    ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
    if not ab.empty:
        fig = px.bar(ab, x='Tipo', y='Engajamento', color='Tamanho', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(lin_f[['Data', 'Título', 'Tag Produto', 'Tipo', 'Tamanho', 'Engajamento', 'Taxa Engajamento (ER)', 'Link']].sort_values('Engajamento', ascending=False),
                 column_config={"Taxa Engajamento (ER)": st.column_config.NumberColumn(format="%.2f%%"), "Link": st.column_config.LinkColumn("🔗 Ver")}, use_container_width=True, hide_index=True)

elif menu == "📧 Mailchimp":
    st.title("Mailchimp: Funil de Retenção")
    ctor_g = mai_f['Cliques_Abs'].sum() / mai_f['Aberturas_Abs'].sum() if mai_f['Aberturas_Abs'].sum() > 0 else 0
    tx_ab_g = mai_f['Aberturas_Abs'].sum() / mai_f['Qtd Enviados'].sum() if mai_f['Qtd Enviados'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Taxa Abertura Global", f_br(tx_ab_g, True))
    c2.metric("CTOR Global", f_br(ctor_g, True))
    c3.metric("Total E-mails", f_br(mai_f['Qtd Enviados'].sum()))

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Funil de Intenção")
        fig_funil = go.Figure(go.Funnel(
            y=['1. Enviados', '2. Aberturas', '3. Cliques'],
            x=[mai_f['Qtd Enviados'].sum(), mai_f['Aberturas_Abs'].sum(), mai_f['Cliques_Abs'].sum()],
            textinfo="value+percent initial", marker={"color": ["#1E293B", "#F97316", "#10B981"]}
        ))
        st.plotly_chart(fig_funil, use_container_width=True)

    with col2:
        st.subheader("Abertura vs CTOR (Desempenho)")
        mai_plot = mai_f.copy()
        mai_plot['Diff CTOR'] = np.where(ctor_g==0, 0, (mai_plot['CTOR'] - ctor_g) / ctor_g)
        mai_plot['Tooltip'] = "CTOR: " + (mai_plot['Diff CTOR']*100).map('{:+.2f}%'.format).str.replace('.', ',') + " da média."

        fig_scat = px.scatter(mai_plot, x='Taxa de Abertura', y='CTOR', size='Qtd Enviados', color='Tag Produto', hover_name='Título',
                              color_discrete_map=CORES_PRODUTOS, custom_data=['Tooltip', 'Ignorados', 'Aberturas_Abs'])
        fig_scat.update_traces(hovertemplate="<b>%{hovertext}</b><br>Abertura: %{x:.2%}<br>CTOR: %{y:.2%}<br>📊 Aberturas: %{customdata[2]}<br><i>%{customdata[0]}</i><extra></extra>")
        fig_scat.add_vline(x=tx_ab_g, line_dash="dot", line_color="red")
        fig_scat.add_hline(y=ctor_g, line_dash="dot", line_color="red")
        st.plotly_chart(fig_scat, use_container_width=True)

    st.dataframe(mai_f[['Data de Envio', 'Título', 'Tag Produto', 'Qtd Enviados', 'Taxa de Abertura', 'CTOR']].sort_values('CTOR', ascending=False),
                 column_config={"Taxa de Abertura": st.column_config.NumberColumn(format="%.2f%%"), "CTOR": st.column_config.NumberColumn(format="%.2f%%")}, use_container_width=True, hide_index=True)

elif menu == "📝 Blog":
    st.title("Blog: Retenção Orgânica")
    tempo_g = blo_f['Tempo da Página'].mean() if not blo_f.empty else 0
    tx_conv_g = blo_f['Clicks'].sum() / blo_f['Views'].sum() if blo_f['Views'].sum() > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Tempo Médio Global", f_br(tempo_g) + "s")
    c2.metric("Conversão Média", f_br(tx_conv_g, True))
    c3.metric("Leituras Orgânicas", f_br(blo_f['Views'].sum()))

    st.markdown("---")
    blo_plot = blo_f.copy()
    blo_plot['Diff Retencao'] = np.where(tempo_g==0, 0, (blo_plot['Tempo da Página'] - tempo_g) / tempo_g)
    blo_plot['Tooltip'] = "Retenção: " + (blo_plot['Diff Retencao']*100).map('{:+.1f}%'.format).str.replace('.', ',') + " da média."

    fig = px.scatter(blo_plot, x='Tempo da Página', y='Taxa Conversão', size='Views', color='Tag Produto', hover_name='Título',
                     color_discrete_map=CORES_PRODUTOS, custom_data=['Tooltip', 'Views'])
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Tempo: %{x}s<br>Conversão: %{y:.2%}<br>👀 Views: %{customdata[1]}<br><i>%{customdata[0]}</i><extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(blo_f[['Data', 'Título', 'Tag Produto', 'Views', 'Tempo da Página', 'Taxa Conversão', 'Link']].sort_values('Views', ascending=False),
                 column_config={"Taxa Conversão": st.column_config.NumberColumn(format="%.2f%%"), "Link": st.column_config.LinkColumn("🔗 Ler")}, use_container_width=True, hide_index=True)

# ==========================================
# 🤖 GUIA EXCLUSIVA: IA & MACHINE LEARNING
# ==========================================
elif menu == "🤖 IA & Machine Learning":
    st.title("🤖 Centro de Inteligência Artificial")
    st.markdown("Algoritmos preditivos, agrupamentos e detecção de anomalias aplicados aos seus dados de marketing.")
    st.markdown("---")

    if over_f.empty:
        st.warning("⚠️ Não há dados suficientes no período ou produto selecionado para gerar modelos de IA.")
    else:
        # 1. Anomaly Detection (Z-Score)
        st.header("1. Radar de Anomalias (Z-Score)")
        st.markdown("A IA analisa a série histórica para separar os conteúdos que **viralizaram** daqueles que foram para a **geladeira** (baixo engajamento anormal).")

        evo = over_f.groupby('Data')['Tração'].sum().reset_index()
        if len(evo) > 2:
            z_scores = stats.zscore(evo['Tração'])
            evo['Z-Score'] = z_scores

            dias_virais = evo[evo['Z-Score'] > 1.0] # Picos acima do desvio
            dias_frios = evo[evo['Z-Score'] < -1.0] # Quedas severas

            col_v, col_f = st.columns(2)
            with col_v:
                st.success(f"🚀 **{len(dias_virais)} Pico(s) Viral(is)** detectado(s).")
                if not dias_virais.empty:
                    # Filtra a lista mestra para mostrar o que gerou o pico
                    lista_virais = lista_f[lista_f['Data'].isin(dias_virais['Data'])].sort_values('Cliques/Tração', ascending=False)
                    st.dataframe(lista_virais[['Data', 'Plataforma', 'Título']], hide_index=True, use_container_width=True)

            with col_f:
                st.error(f"🧊 **{len(dias_frios)} Dia(s) Frio(s)** detectado(s) (Abaixo da média).")
                if not dias_frios.empty:
                    lista_frios = lista_f[lista_f['Data'].isin(dias_frios['Data'])].sort_values('Cliques/Tração', ascending=True)
                    st.dataframe(lista_frios[['Data', 'Plataforma', 'Título']], hide_index=True, use_container_width=True)
        else:
            st.info("Volume de dias insuficiente para calcular anomalias estáticas. Expanda o filtro de datas.")

        st.markdown("---")

        # 2. KMeans (Clusterização de E-mail)
        st.header("2. Clusterização de Audiência (K-Means)")
        st.markdown("O algoritmo K-Means agrupou automaticamente suas campanhas de **Mailchimp** de acordo com o padrão de comportamento dos MSPs/Revendas.")

        if len(mai_f) > 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            X = mai_f[['Taxa de Abertura', 'CTOR']].fillna(0)
            mai_f['Cluster (IA)'] = kmeans.fit_predict(X)
            # Ordenando os clusters para garantir que o 0 seja o pior e o 2 o melhor, ou vice-versa
            centroides = kmeans.cluster_centers_
            # Cria um mapeamento baseado na soma (Abertura + CTOR)
            soma_centroides = centroides.sum(axis=1)
            ranking_clusters = soma_centroides.argsort()
            mapa = {ranking_clusters[0]: 'Baixa Atenção ⚠️', ranking_clusters[1]: 'Média 📉', ranking_clusters[2]: 'Estrela 🚀'}
            mai_f['Classificação IA'] = mai_f['Cluster (IA)'].map(mapa)

            fig_km = px.scatter(
                mai_f, x='Taxa de Abertura', y='CTOR', color='Classificação IA', size='Qtd Enviados', hover_name='Título',
                color_discrete_map={'Estrela 🚀': '#10B981', 'Média 📉': '#F59E0B', 'Baixa Atenção ⚠️': '#DC2626'}
            )
            fig_km.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
            st.plotly_chart(fig_km, use_container_width=True)
        else:
            st.info("E-mails insuficientes para treinar o K-Means. O modelo precisa de pelo menos 4 campanhas diferentes.")

        st.markdown("---")

        # 3. OLS Regression (Regressão no Blog)
        st.header("3. Previsibilidade de Conversão (OLS Regression)")
        st.markdown("Traçamos uma Regressão Linear sobre o tráfego do **Blog**. O objetivo é provar estatisticamente se a retenção (tempo lendo) resulta diretamente em conversão (cliques).")

        if len(blo_f) > 2:
            fig_ols = px.scatter(
                blo_f, x='Tempo da Página', y='Taxa Conversão', size='Views', color='Tag Produto', hover_name='Título',
                color_discrete_map=CORES_PRODUTOS, trendline="ols", trendline_scope="overall"
            )
            fig_ols.update_traces(hovertemplate="Tempo: %{x}s<br>Conversão: %{y:.2%}<extra></extra>")
            st.plotly_chart(fig_ols, use_container_width=True)
        else:
            st.info("Dados insuficientes do Blog para traçar a Regressão Linear.")

        st.markdown("---")

        # 4. Feature Matrix (LinkedIn)
        st.header("4. Matriz de Densidade (LinkedIn)")
        st.markdown("Mapa de calor que cruza variáveis qualitativas (Tamanho do Texto vs Estilo) para achar a **Fórmula de Copy** que o algoritmo do LinkedIn mais entrega.")

        ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
        if not ab.empty:
            fig_hm = px.density_heatmap(ab, x="Tipo", y="Tamanho", z="Engajamento", text_auto=".1f", color_continuous_scale="Blues")
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Sem dados do LinkedIn para renderizar a matriz.")
