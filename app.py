import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
import datetime

st.set_page_config(
    page_title="AIQON | Marketing Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 TEMA VISUAL — Variáveis Nativas (Light/Dark Automático)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--faded-text-10);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Tabelas */
[data-testid="stDataFrame"] { border: 1px solid var(--faded-text-10); border-radius: 12px; overflow: hidden; }

/* Cards informativos */
.info-card {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--faded-text-10);
    border-left: 3px solid #0094A4;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.info-card p { font-size: 0.88rem !important; margin: 0; color: var(--text-color); }
.info-card strong { color: var(--text-color); }

/* Seção explicativa */
.explain-box {
    background-color: var(--secondary-background-color);
    border: 1px solid var(--faded-text-10);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
}
.explain-box h4 { color: #0094A4 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.explain-box p { font-size: 0.9rem !important; line-height: 1.6; margin: 0; color: var(--text-color); }

/* Plotly charts */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🔧 CONFIGURAÇÕES
# ============================================================
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
    'Institucional / Outros': '#808080', # Cinza neutro para light/dark
    'Syxsense': '#A855F7'
}

def f_br(num, is_pct=False):
    if pd.isna(num): return "0"
    if is_pct:
        return f"{num * 100:.2f}%".replace('.', ',')
    else:
        return f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# ============================================================
# 🏷️ SISTEMA DE TAGS AUTOMÁTICO
# ============================================================
TAGS_TIPO = {
    'Produto': ['action1', 'netwrix', '42crunch', 'ox ', 'cynet', 'easy inventory',
                'keepit', 'grip', 'manage engine', 'wallarm', 'syxsense', 'lançamento', 'release', 'solução'],
    'Notícia': ['alerta', 'cve-', 'vulnerabilidade', 'ataque', 'ransomware', 'breach',
                'vazamento', 'news', 'urgente', 'incidente', 'crise'],
    'Conceito': ['o que é', 'como funciona', 'guia', 'entenda', 'conceito', 'introdução',
                 'fundamentos', 'overview', 'explicando', 'por que'],
    'Campanha / Ebook': ['ebook', 'download', 'checklist', 'whitepaper', 'webinar',
                         'evento', 'live', 'inscreva', 'acesse grátis', 'material'],
    'Comercial': ['parceria', 'cliente', 'case', 'resultado', 'roi', 'demonstração',
                  'demo', 'contrate', 'venda', 'oferta'],
}

def detectar_tags(texto):
    texto_l = str(texto).lower()
    tags = []
    for tag, palavras in TAGS_TIPO.items():
        if any(p in texto_l for p in palavras):
            tags.append(tag)
    if not tags:
        tags.append('Editorial / Educativo')
    return tags

def tags_para_str(tags_list):
    if isinstance(tags_list, list):
        return ', '.join(tags_list)
    return str(tags_list)

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

# ============================================================
# 🧠 MOTOR DE DADOS
# ============================================================
@st.cache_data(ttl=60)
def carregar_dados():
    try:
        lin = pd.read_csv('dataset/linkedin_clean.csv')
        blo = pd.read_csv('dataset/blog_clean.csv')
        mai = pd.read_csv('dataset/mailchimp_clean.csv')
    except:
        hoje = pd.Timestamp.today()
        lin = pd.DataFrame({'Data da postagem': [hoje, hoje - pd.Timedelta(days=1), hoje - pd.Timedelta(days=1)], 'Texto': ['O que é a nova solução da Netwrix?', 'Baixe nosso ebook sobre Action1', 'O que é a nova solução da Netwrix?'], 'Link da Postagem': ['link1', 'link2', 'link1'], 'Curtidas': [45, 120, 20], 'Comentários': [5, 15, 2], 'Shares': [2, 10, 1], 'Seguidores': [5000, 5000, 5000]})
        blo = pd.DataFrame({'Data': [hoje, hoje - pd.Timedelta(days=2)], 'URL': ['https://aiqon.com.br/blog/guia-netwrix', 'https://aiqon.com.br/blog/cynet-news'], 'Views': [350, 150], 'Clicks': [25, 5], 'Tempo da Página': [210, 90]})
        mai = pd.DataFrame({'Data de Envio': [hoje - pd.Timedelta(days=3), hoje - pd.Timedelta(days=4)], 'Título': ['Netwrix Webinar', 'Guia Action1'], 'Qtd Enviados': [10000, 8000], 'Taxa de Abertura': [0.28, 0.15], 'Clicks': [0.05, 0.01]})

    lin['Data'] = pd.to_datetime(lin['Data da postagem'], errors='coerce').dt.date
    blo['Data'] = pd.to_datetime(blo['Data'], errors='coerce').dt.date
    mai['Data de Envio'] = pd.to_datetime(mai['Data de Envio'], errors='coerce').dt.date

    # ---------- MAILCHIMP ----------
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    mai['Tags Tipo'] = mai['Título'].apply(lambda x: tags_para_str(detectar_tags(x)))
    mai['Aberturas_Abs'] = (mai['Qtd Enviados'] * mai['Taxa de Abertura']).fillna(0).round().astype(int)
    mai['Cliques_Abs'] = (mai['Qtd Enviados'] * mai['Clicks']).fillna(0).round().astype(int)

    mai_grp = mai.groupby(['Título', 'Tag Produto', 'Tags Tipo']).agg({
        'Qtd Enviados': 'sum', 'Aberturas_Abs': 'sum', 'Cliques_Abs': 'sum', 'Data de Envio': 'max'
    }).reset_index()
    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados'] == 0, 0, mai_grp['Aberturas_Abs'] / mai_grp['Qtd Enviados'])
    mai_grp['CTOR'] = np.where(mai_grp['Aberturas_Abs'] == 0, 0, mai_grp['Cliques_Abs'] / mai_grp['Aberturas_Abs'])
    mai_grp['Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas_Abs']
    mai_grp['Plataforma'] = 'Mailchimp'
    mai_grp['Link'] = 'N/A'

    # ---------- BLOG ----------
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Tags Tipo'] = blo['URL'].apply(lambda x: tags_para_str(detectar_tags(x)))
    blo['Título'] = (blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False).str.replace("/", "", regex=False).str.replace("-", " ").str.title())

    blo_grp = blo.groupby(['URL', 'Título', 'Tag Produto', 'Tags Tipo']).agg({
        'Views': 'sum', 'Clicks': 'sum', 'Tempo da Página': 'mean', 'Data': 'max'
    }).reset_index()
    blo_grp['Taxa Conversão'] = np.where(blo_grp['Views'] == 0, 0, blo_grp['Clicks'] / blo_grp['Views'])
    blo_grp['Plataforma'] = 'Blog'
    blo_grp['Link'] = blo_grp['URL']

    # ---------- LINKEDIN ----------
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tags Tipo'] = lin['Texto'].apply(lambda x: tags_para_str(detectar_tags(x)))
    lin['Tamanho'] = np.where(lin['Texto'].str.len() < 500, 'Curto', np.where(lin['Texto'].str.len() < 1200, 'Médio', 'Longo'))
    lin['Tipo'] = np.where(lin['Texto'].str.contains('parceria|solução|contrat|demo|demonstr', case=False, na=False), 'Comercial/Venda', 'Editorial/Educativo')
    lin['Engajamento'] = (lin['Curtidas'].fillna(0) + lin['Comentários'].fillna(0) + lin['Shares'].fillna(0))

    lin_raw = lin.copy()
    lin_raw = lin_raw.sort_values(['Link da Postagem', 'Data'])
    lin_raw['Engajamento Acum'] = lin_raw.groupby('Link da Postagem')['Engajamento'].cumsum()
    lin_raw['Medicao_Nr'] = lin_raw.groupby('Link da Postagem').cumcount() + 1

    lin_grp = lin.groupby(['Link da Postagem', 'Título', 'Tag Produto', 'Tags Tipo', 'Tamanho', 'Tipo']).agg({
        'Seguidores': 'max', 'Data': 'max', 'Engajamento': 'sum'
    }).reset_index()
    lin_grp['Taxa Engajamento (ER)'] = np.where(lin_grp['Seguidores'] == 0, 0, lin_grp['Engajamento'] / lin_grp['Seguidores'])
    lin_grp['Plataforma'] = 'LinkedIn'
    lin_grp['Link'] = lin_grp['Link da Postagem']

    def detectar_padrao(grupo):
        if len(grupo) < 2: return 'Dado Único'
        vals = grupo.sort_values('Data')['Engajamento'].values
        diffs = np.diff(vals)
        if diffs[0] > 0 and all(d <= 0 for d in diffs[1:]): return 'Spike Inicial 🚀'
        elif all(d >= 0 for d in diffs): return 'Crescimento Constante 📈'
        elif diffs[-1] < 0 and diffs[0] < 0: return 'Queda Rápida 📉'
        else: return 'Variado 〰️'

    padroes = lin_raw.groupby('Link da Postagem').apply(detectar_padrao, include_groups=False).reset_index()
    padroes.columns = ['Link da Postagem', 'Padrão']
    lin_grp = lin_grp.merge(padroes, on='Link da Postagem', how='left')
    lin_grp['Padrão'] = lin_grp['Padrão'].fillna('Dado Único')

    # ---------- OVERVIEW ----------
    r_lin = lin_grp.groupby(['Data', 'Tag Produto'])['Engajamento'].sum().reset_index().rename(columns={'Engajamento': 'Tração'})
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index().rename(columns={'Views': 'Tração'})
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas_Abs'].sum().reset_index().rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Tração'})

    over = pd.concat([r_lin, r_blo, r_mai]).groupby(['Data', 'Tag Produto'])['Tração'].sum().reset_index()
    over = over.sort_values('Data')

    lista = pd.concat([
        lin_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Engajamento']].rename(columns={'Engajamento': 'Cliques/Tração'}),
        blo_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Views']].rename(columns={'Views': 'Cliques/Tração'}),
        mai_grp[['Data de Envio', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Aberturas_Abs']].rename(columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Cliques/Tração'})
    ]).sort_values('Data', ascending=False)

    return over, lin_grp, lin_raw, blo_grp, mai_grp, lista

# ============================================================
# CARREGA DADOS E HELPER EXPLAIN
# ============================================================
with st.spinner("Carregando dados..."):
    df_over, df_lin, df_lin_raw, df_blo, df_mai, df_lista = carregar_dados()

def explain(titulo, descricao):
    st.markdown(f"""
    <div class="explain-box">
        <h4>{titulo}</h4>
        <p>{descricao}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
LOGO_B64 = "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMiIgZGF0YS1uYW1lPSJMYXllciAyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODcuNDQgMTY2LjA1Ij4KICA8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiAjMDA5NGE0OwogICAgICB9CgogICAgICAuY2xzLTIgewogICAgICAgIGZpbGw6IGdyYXk7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxnIGlkPSJMYXllcl8yLTIiIGRhdGEtbmFtZT0iTGF5ZXIgMiI+CiAgICA8Zz4KICAgICAgPGc+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMzc5Ljg4LDM2LjY1Yy01LjMyLTUuMzItMTEuOTMtNy44MS0xOS40OC03Ljgxcy0xNC4wOCwyLjQ5LTE5LjQsNy45Yy01LjQ5LDUuNDktNy44MSwxMi4zNi03LjgxLDIwdjQxLjExaDEwLjgydi00MC41MWMwLTQuODksMS4zNy05LjM2LDQuNzItMTIuOTYsMy4xOC0zLjM1LDYuOTUtNC45OCwxMS41OS00Ljk4czguNDEsMS41NSwxMS41OSw0Ljk4YzMuNDMsMy42MSw0LjcyLDguMDcsNC43MiwxMi45NnY0MC41MWgxMC44MnYtNDEuMTFjLjE3LTcuNjQtMi4xNS0xNC41OS03LjY0LTIwLjA5LDAsMCwuMDksMCwuMDksMFoiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yOTEuMywyNy4xMmMtMTkuNTcsMC0zNS42MiwxNS45Ny0zNS42MiwzNS42MnMxNS45NywzNS42MiwzNS42MiwzNS42MiwzNS42Mi0xNS45NywzNS42Mi0zNS42Mi0xNS45Ny0zNS42Mi0zNS42Mi0zNS42MlpNMjkxLjMsODYuODZjLTEzLjMsMC0yNC4wMy0xMC44Mi0yNC4wMy0yNC4wM3MxMC44Mi0yNC4wMywyNC4wMy0yNC4wMywyNC4wMywxMC44MiwyNC4wMywyNC4wMy0xMC44MiwyNC4wMy0yNC4wMywyNC4wM1oiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yMTYuOCwyOC44NGMtOS4xLDAtMTYuOTEsMy4zNS0yMy4wOSw5Ljk2LTYuNTIsNi44Ny05LjI3LDE1LjE5LTkuMjcsMjQuNTVzMi44MywxNy43Nyw5LjM2LDI0LjYzYzYuMjcsNi41MiwxNC4wOCw5Ljg3LDIzLjA5LDkuODdzNi42MS0uNDMsOS43LTEuNTV2LTExLjQyYy0uMjYuMTctLjYuMjYtLjg2LjQzLTIuODMsMS4zNy01Ljg0LDEuODktOC45MywxLjg5LTYuMDEsMC0xMC43My0yLjU4LTE0Ljc2LTYuODctNC41NS00LjcyLTYuNTItMTAuNTYtNi41Mi0xNy4wOHMyLjA2LTEyLjM2LDYuNTItMTcuMDhjNC4wMy00LjI5LDguODQtNi44NywxNC43Ni02Ljg3czExLjMzLDIuMjMsMTUuNDUsNi45NWM0LjI5LDQuODksNi4xOCwxMC41Niw2LjE4LDE3djYyLjE0aDEwLjgydi02Mi4yM2MwLTkuMzYtMi44My0xNy43Ny05LjM2LTI0LjYzLTYuMDktNi41Mi0xMy45MS05Ljg3LTIzLTkuODdoMGwtLjA5LjE3aDBaIi8+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMTY1LjMsMjguODRoMTAuODJ2NjkuMDFoLTEwLjgyVjI4Ljg0WiIvPgogICAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTEyNC43LDI4Ljg0Yy05LjEsMC0xNi45MSwzLjM1LTIzLjA5LDkuOTYtNi40NCw2Ljg3LTkuMjcsMTUuMTktOS4yNywyNC41NXMyLjgzLDE3Ljc3LDkuMzYsMjQuNjNjNi4yNyw2LjUyLDE0LjA4LDkuODcsMjMuMDksOS44N3M2LjUyLS40Myw5LjYxLTEuNTV2LTExLjMzYy0uMjYuMTctLjUyLjI2LS43Ny40My0yLjgzLDEuMzctNS44NCwxLjg5LTguOTMsMS44OS02LjAxLDAtMTAuNzMtMi41OC0xNC43Ni02Ljg3LTQuNTUtNC43Mi02LjUyLTEwLjU2LTYuNTItMTcuMDhzMS45Ny0xMi4yNyw2LjUyLTE3LjA4YzQuMDMtNC4yOSw4Ljg0LTYuODcsMTQuNzYtNi44N3MxMS4zMywyLjMsMTUuNDUsNi45NWM0LjIxLDQuODksNi4xOCwxMC41Niw2LjE4LDE3djEwLjgyaDB2MTQuODVoMHY4Ljc2aDEwLjgydi0zNC41MWMwLTkuMzYtMi44My0xNy43Ny05LjM2LTI0LjYzLTYuMTgtNi41Mi0xMy45OS05Ljg3LTIzLTkuODdoMGwtLjA5LjA5aDBaIi8+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNOS41MSwzNS4wMmMtNS4wNiwxNy40MiwyNi40NCwzMC4zOSwzOC44LDUxLjA3LDEyLjEsMjAuMzQtMS43MiwzMS4wNy0xMS44NSwzOS40OCw0LjAzLTEzLjMtMTIuMjctMjEuOC0yNi4zNS0zNy41MS0xMy45OS0xNS4yOC0xMi43OS0zNi4yMi0uNTItNTMuMDUsMCwwLS4wOSwwLS4wOSwwWiIvPgogICAgICA8L2c+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTQ0Ljg3LDBjLTYuMTgsMjEuMzcsMzYuOTEsMzAuMjEsMzYuNjUsNjEuOCwwLDkuODctNC44MSwxOC43MS0xMS42NywyOS4xLDUuODQtMTEuODUtMTQuMTYtMjYuNTItMjguNDEtNDIuMDYtMTMuOTEtMTUuMjgtMjEuOC0zMi42MiwzLjQzLTQ4Ljg0WiIvPgogICAgPC9nPgo8L3N2Zz4="

st.sidebar.image(LOGO_B64, width=180)
st.sidebar.markdown("### Marketing Intelligence")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Navegação", [
    "🌐 Overview Geral",
    "💼 LinkedIn — Engajamento",
    "📧 E-mail Marketing",
    "📝 Blog & SEO",
    "🏷️ Performance por Tag",
    "🤖 IA & Modelos Preditivos"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros Globais**")

todas_datas = pd.Series(df_over['Data'].dropna().tolist() + df_lin['Data'].dropna().tolist() + df_blo['Data'].dropna().tolist() + df_mai['Data de Envio'].dropna().tolist())

if not todas_datas.empty:
    min_d, max_d = todas_datas.min(), todas_datas.max()
else:
    min_d = max_d = datetime.date.today()

datas = st.sidebar.date_input("Período", [min_d, max_d], min_value=min_d, max_value=max_d, format="DD/MM/YYYY")

prod_disp = sorted(df_over['Tag Produto'].unique().tolist())
filtro_prod = st.sidebar.multiselect("Produto / Marca", prod_disp, default=prod_disp)

tags_disp = ['Produto', 'Notícia', 'Conceito', 'Campanha / Ebook', 'Comercial', 'Editorial / Educativo']
filtro_tag = st.sidebar.multiselect("Tipo de Conteúdo", tags_disp, default=tags_disp)

if len(datas) == 2:
    d0, d1 = datas[0], datas[1]
    def in_date(df, col='Data'): return (df[col] >= d0) & (df[col] <= d1)
    def tag_match(df): return df['Tags Tipo'].apply(lambda t: any(f in str(t) for f in filtro_tag))

    over_f = df_over[in_date(df_over) & df_over['Tag Produto'].isin(filtro_prod)]
    lin_f  = df_lin[in_date(df_lin) & df_lin['Tag Produto'].isin(filtro_prod) & tag_match(df_lin)]
    lin_raw_f = df_lin_raw[in_date(df_lin_raw) & df_lin_raw['Tag Produto'].isin(filtro_prod)]
    blo_f  = df_blo[in_date(df_blo) & df_blo['Tag Produto'].isin(filtro_prod) & tag_match(df_blo)]
    mai_f  = df_mai[in_date(df_mai, 'Data de Envio') & df_mai['Tag Produto'].isin(filtro_prod) & tag_match(df_mai)]
    lista_f = df_lista[in_date(df_lista) & df_lista['Tag Produto'].isin(filtro_prod)]
else:
    over_f = lin_f = lin_raw_f = blo_f = mai_f = lista_f = pd.DataFrame(columns=df_over.columns)

# ============================================================
# 🌐 1. OVERVIEW GERAL
# ============================================================
if menu == "🌐 Overview Geral":
    st.title("Overview de Marketing")
    explain("O que é este painel?", "Visão consolidada de toda a presença digital: LinkedIn, Blog e E-mail. O indicador <strong>Tração</strong> soma engajamentos do LinkedIn, visualizações do Blog e aberturas de e-mail.")

    c1, c2, c3, c4 = st.columns(4)
    total_tracao = over_f['Tração'].sum()
    pico = over_f.groupby('Data')['Tração'].sum().max() if not over_f.empty else 0
    media_dia = over_f.groupby('Data')['Tração'].sum().mean() if not over_f.empty else 0
    n_produtos = len(over_f['Tag Produto'].unique())

    c1.metric("Tração Total", f_br(total_tracao), help="Soma de engajamentos LI + views blog + aberturas email")
    c2.metric("Pico Diário", f_br(pico))
    c3.metric("Média Diária", f_br(media_dia))
    c4.metric("Produtos Ativos", str(n_produtos))

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Evolução de Tração por Plataforma")
        explain("Como ler este gráfico", "A linha tracejada é a média do período — dias acima dela foram acima do esperado.")
        evo = over_f.groupby('Data')['Tração'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=evo['Data'], y=evo['Tração'], marker_color='#0094A4', name='Tração Diária', hovertemplate='<b>%{x}</b><br>Tração: %{y:,.0f}<extra></extra>'))
        if not evo.empty and len(evo) > 1:
            media = evo['Tração'].mean()
            fig.add_hline(y=media, line_dash='dash', line_color='#F97316', annotation_text=f'Média: {f_br(media)}', annotation_font_color='#F97316')
        st.plotly_chart(fig, width="stretch", theme="streamlit")

    with col2:
        st.markdown("#### Tração por Produto")
        explain("Como ler", "Mostra quais produtos geraram mais atenção no período. Use para decidir onde intensificar ou reduzir esforços.")
        rank = over_f.groupby('Tag Produto')['Tração'].sum().reset_index().sort_values('Tração', ascending=True)
        rank = rank[rank['Tag Produto'] != 'Institucional / Outros']
        if not rank.empty:
            fig2 = px.bar(rank, y='Tag Produto', x='Tração', orientation='h', color='Tag Produto', color_discrete_map=CORES_PRODUTOS)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, width="stretch", theme="streamlit")

    st.markdown("#### Distribuição por Canal")
    explain("Proporção de canais", "Entenda quanto cada canal contribui para a tração total.")
    lin_total = lin_f['Engajamento'].sum() if not lin_f.empty else 0
    blo_total = blo_f['Views'].sum() if not blo_f.empty else 0
    mai_total = mai_f['Aberturas_Abs'].sum() if not mai_f.empty else 0
    canais = pd.DataFrame({'Canal': ['LinkedIn', 'Blog', 'E-mail'], 'Volume': [lin_total, blo_total, mai_total], 'Cor': ['#0A66C2', '#10B981', '#F97316']})
    
    if canais['Volume'].sum() > 0:
        fig_pie = go.Figure(go.Pie(labels=canais['Canal'], values=canais['Volume'], hole=0.6, marker=dict(colors=['#0A66C2', '#10B981', '#F97316']), textinfo='label+percent'))
        fig_pie.update_layout(height=280)
        fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<extra></extra>')
        st.plotly_chart(fig_pie, width="stretch", theme="streamlit")

    st.markdown("#### Lista Mestra de Conteúdo")
    st.dataframe(lista_f[['Data', 'Plataforma', 'Título', 'Tag Produto', 'Tags Tipo', 'Cliques/Tração', 'Link']], column_config={"Data": st.column_config.DateColumn(format="DD/MM/YYYY"), "Link": st.column_config.LinkColumn("🔗 Abrir"), "Cliques/Tração": st.column_config.NumberColumn(format="%d")}, width="stretch", hide_index=True)

# ============================================================
# 💼 2. LINKEDIN
# ============================================================
elif menu == "💼 LinkedIn — Engajamento":
    st.title("LinkedIn — Análise de Engajamento")
    explain("O que analisamos aqui?", "Engajamento = Curtidas + Comentários + Shares. O <strong>ER (Engagement Rate)</strong> divide o engajamento pelo número de seguidores.")

    c1, c2, c3, c4 = st.columns(4)
    media_eng = lin_f['Engajamento'].mean() if not lin_f.empty else 0
    er_global = (lin_f['Engajamento'].sum() / lin_f['Seguidores'].sum() if not lin_f.empty and lin_f['Seguidores'].sum() > 0 else 0)
    n_posts = len(lin_f)
    top_post = lin_f['Engajamento'].max() if not lin_f.empty else 0

    c1.metric("Média Engajamento/Post", f_br(media_eng))
    c2.metric("ER Global", f_br(er_global, True), help="Engajamento Total ÷ Seguidores Máx")
    c3.metric("Posts Analisados", str(n_posts))
    c4.metric("Maior Engajamento", f_br(top_post))

    st.markdown("---")
    st.markdown("#### Curva de Engajamento ao Longo do Tempo")
    posts_multi = lin_raw_f.groupby('Link da Postagem').filter(lambda x: len(x) > 1)
    if not posts_multi.empty:
        top_posts = (lin_f.nlargest(10, 'Engajamento')['Link da Postagem'].tolist())
        curva_df = posts_multi[posts_multi['Link da Postagem'].isin(top_posts)]
        fig_curva = px.line(curva_df.sort_values('Data'), x='Data', y='Engajamento', color='Título', markers=True, line_shape='spline', title='Top 10 Posts — Engajamento por Medição')
        fig_curva.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Data: %{x}<br>Engajamento: %{y}<extra></extra>')
        st.plotly_chart(fig_curva, width="stretch", theme="streamlit")
    else:
        st.markdown("""
        <div class="info-card">
        <p>⏳ <strong>Dados de medição única detectados.</strong> Veja a evolução temporal agregada por semana.</p>
        </div>
        """, unsafe_allow_html=True)
        lin_temporal = lin_f.copy()
        lin_temporal['Semana'] = pd.to_datetime(lin_temporal['Data']).dt.to_period('W').astype(str)
        semanal = lin_temporal.groupby('Semana')['Engajamento'].mean().reset_index()
        fig_sem = px.line(semanal, x='Semana', y='Engajamento', markers=True, title='Engajamento Médio por Semana')
        st.plotly_chart(fig_sem, width="stretch", theme="streamlit")

    st.markdown("#### Padrões de Comportamento dos Posts")
    if 'Padrão' in lin_f.columns and not lin_f.empty:
        pad_count = lin_f['Padrão'].value_counts().reset_index()
        pad_count.columns = ['Padrão', 'Posts']
        fig_pad = px.bar(pad_count, x='Padrão', y='Posts', color='Padrão', color_discrete_sequence=['#10B981', '#0094A4', '#DC2626', '#F59E0B'])
        fig_pad.update_layout(showlegend=False)
        st.plotly_chart(fig_pad, width="stretch", theme="streamlit")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Engajamento por Tipo × Tamanho")
        ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
        if not ab.empty:
            fig_hm = px.density_heatmap(ab, x='Tipo', y='Tamanho', z='Engajamento', text_auto='.1f', color_continuous_scale='Teal')
            st.plotly_chart(fig_hm, width="stretch", theme="streamlit")

    with col2:
        st.markdown("#### Top Tags de Conteúdo")
        from collections import Counter
        todas_tags = []
        for row in lin_f['Tags Tipo'].dropna():
            todas_tags.extend([t.strip() for t in str(row).split(',')])
        if todas_tags:
            tag_df = pd.DataFrame(Counter(todas_tags).most_common(), columns=['Tag', 'Posts'])
            fig_tags = px.bar(tag_df, x='Posts', y='Tag', orientation='h', color_discrete_sequence=['#0094A4'])
            fig_tags.update_layout(showlegend=False)
            st.plotly_chart(fig_tags, width="stretch", theme="streamlit")

    st.markdown("#### Todos os Posts")
    cols_show = ['Data', 'Título', 'Tag Produto', 'Tags Tipo', 'Tipo', 'Tamanho', 'Engajamento', 'Taxa Engajamento (ER)', 'Link']
    cols_show = [c for c in cols_show if c in lin_f.columns]
    st.dataframe(lin_f[cols_show].sort_values('Engajamento', ascending=False), column_config={"Data": st.column_config.DateColumn(format="DD/MM/YYYY"), "Taxa Engajamento (ER)": st.column_config.NumberColumn(format="%.4f"), "Link": st.column_config.LinkColumn("🔗 Ver Post"), "Engajamento": st.column_config.NumberColumn(format="%d")}, width="stretch", hide_index=True)

# ============================================================
# 📧 3. E-MAIL MARKETING
# ============================================================
elif menu == "📧 E-mail Marketing":
    st.title("E-mail Marketing — Funil e Conversão")
    explain("Métricas principais", "<strong>Taxa de Abertura</strong>: % que abriram o e-mail. <strong>CTOR</strong>: % dos que abriram e clicaram.")

    ctor_g = (mai_f['Cliques_Abs'].sum() / mai_f['Aberturas_Abs'].sum() if not mai_f.empty and mai_f['Aberturas_Abs'].sum() > 0 else 0)
    tx_ab_g = (mai_f['Aberturas_Abs'].sum() / mai_f['Qtd Enviados'].sum() if not mai_f.empty and mai_f['Qtd Enviados'].sum() > 0 else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taxa de Abertura", f_br(tx_ab_g, True))
    c2.metric("CTOR Global", f_br(ctor_g, True))
    c3.metric("Total Enviados", f_br(mai_f['Qtd Enviados'].sum()))
    c4.metric("Campanhas", str(len(mai_f)))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Funil de E-mail")
        if not mai_f.empty:
            fig_funil = go.Figure(go.Funnel(y=['Enviados', 'Aberturas', 'Cliques'], x=[mai_f['Qtd Enviados'].sum(), mai_f['Aberturas_Abs'].sum(), mai_f['Cliques_Abs'].sum()], textinfo='value+percent initial', marker={'color': ['#1E293B', '#0094A4', '#10B981']}))
            fig_funil.update_layout(height=320)
            st.plotly_chart(fig_funil, width="stretch", theme="streamlit")

    with col2:
        st.markdown("#### Quadrante de Desempenho: Abertura × CTOR")
        if not mai_f.empty:
            mai_plot = mai_f.copy()
            def quadrante(row):
                if row['Taxa de Abertura'] >= tx_ab_g and row['CTOR'] >= ctor_g: return '🌟 Estrela'
                elif row['Taxa de Abertura'] >= tx_ab_g and row['CTOR'] < ctor_g: return '📣 Bom Assunto'
                elif row['Taxa de Abertura'] < tx_ab_g and row['CTOR'] >= ctor_g: return '💎 Nicho Quente'
                else: return '⚠️ Revisar'

            mai_plot['Quadrante'] = mai_plot.apply(quadrante, axis=1)
            CORES_Q = {'🌟 Estrela': '#10B981', '📣 Bom Assunto': '#F59E0B', '💎 Nicho Quente': '#0094A4', '⚠️ Revisar': '#DC2626'}

            fig_scat = px.scatter(mai_plot, x='Taxa de Abertura', y='CTOR', size='Qtd Enviados', color='Quadrante', hover_name='Título', color_discrete_map=CORES_Q, custom_data=['Aberturas_Abs', 'Cliques_Abs', 'Qtd Enviados'])
            fig_scat.update_traces(hovertemplate=("<b>%{hovertext}</b><br>Abertura: %{x:.1%}<br>CTOR: %{y:.1%}<br>Enviados: %{customdata[2]:,.0f}<br>Aberturas: %{customdata[0]:,.0f} | Cliques: %{customdata[1]:,.0f}<extra></extra>"))
            fig_scat.add_vline(x=tx_ab_g, line_dash='dot', line_color='#808080', annotation_text='Média Abertura')
            fig_scat.add_hline(y=ctor_g, line_dash='dot', line_color='#808080', annotation_text='Média CTOR')
            fig_scat.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%', height=360)
            st.plotly_chart(fig_scat, width="stretch", theme="streamlit")

    st.markdown("#### Todas as Campanhas")
    if not mai_f.empty and 'Quadrante' in mai_plot.columns:
        tabela_mai = mai_f.merge(mai_plot[['Título', 'Quadrante']], on='Título', how='left')
        st.dataframe(tabela_mai[['Data de Envio', 'Título', 'Tag Produto', 'Tags Tipo', 'Qtd Enviados', 'Aberturas_Abs', 'Cliques_Abs', 'Taxa de Abertura', 'CTOR', 'Quadrante']].sort_values('CTOR', ascending=False), column_config={"Data de Envio": st.column_config.DateColumn(format="DD/MM/YYYY"), "Taxa de Abertura": st.column_config.NumberColumn(format="%.2f%%"), "CTOR": st.column_config.NumberColumn(format="%.2f%%"), "Aberturas_Abs": st.column_config.NumberColumn("Aberturas", format="%d"), "Cliques_Abs": st.column_config.NumberColumn("Cliques", format="%d")}, width="stretch", hide_index=True)

# ============================================================
# 📝 4. BLOG & SEO
# ============================================================
elif menu == "📝 Blog & SEO":
    st.title("Blog — Retenção, Conversão e SEO")
    explain("Métricas do Blog", "<strong>Views</strong>: total de visualizações. <strong>Tempo na Página</strong>: tempo médio de leitura. <strong>Taxa de Conversão</strong>: % de leitores que clicaram em CTAs.")

    tempo_g = blo_f['Tempo da Página'].mean() if not blo_f.empty else 0
    tx_conv_g = (blo_f['Clicks'].sum() / blo_f['Views'].sum() if not blo_f.empty and blo_f['Views'].sum() > 0 else 0)
    total_views = blo_f['Views'].sum() if not blo_f.empty else 0
    n_artigos = len(blo_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tempo Médio na Página", f"{f_br(tempo_g)}s")
    c2.metric("Taxa de Conversão Média", f_br(tx_conv_g, True))
    c3.metric("Visualizações Totais", f_br(total_views))
    c4.metric("Artigos Monitorados", str(n_artigos))

    st.markdown("#### Quadrante: Retenção × Conversão")
    if not blo_f.empty:
        blo_plot = blo_f.copy()
        def q_blog(row):
            if row['Tempo da Página'] >= tempo_g and row['Taxa Conversão'] >= tx_conv_g: return '🌟 Magnetizador'
            elif row['Tempo da Página'] >= tempo_g and row['Taxa Conversão'] < tx_conv_g: return '📖 Leitura Profunda'
            elif row['Tempo da Página'] < tempo_g and row['Taxa Conversão'] >= tx_conv_g: return '⚡ Conversor Rápido'
            else: return '🔧 Revisar'

        blo_plot['Quadrante'] = blo_plot.apply(q_blog, axis=1)
        CORES_QB = {'🌟 Magnetizador': '#10B981', '📖 Leitura Profunda': '#0094A4', '⚡ Conversor Rápido': '#F59E0B', '🔧 Revisar': '#DC2626'}

        fig_blo = px.scatter(blo_plot, x='Tempo da Página', y='Taxa Conversão', size='Views', color='Quadrante', hover_name='Título', color_discrete_map=CORES_QB, custom_data=['Views', 'Tags Tipo'])
        fig_blo.update_traces(hovertemplate=("<b>%{hovertext}</b><br>Tempo: %{x:.0f}s<br>Conversão: %{y:.1%}<br>Views: %{customdata[0]:,.0f}<br>Tag: %{customdata[1]}<extra></extra>"))
        fig_blo.add_vline(x=tempo_g, line_dash='dot', line_color='#808080', annotation_text='Média Tempo')
        fig_blo.add_hline(y=tx_conv_g, line_dash='dot', line_color='#808080', annotation_text='Média Conv.')
        fig_blo.update_layout(yaxis_tickformat='.1%', height=420)
        st.plotly_chart(fig_blo, width="stretch", theme="streamlit")

    st.markdown("---")
    st.markdown("#### Checklist de SEO e Qualidade de Conteúdo")
    seo_items = [
        ("SEO Score Verde no Plugin", "Todo artigo deve atingir score verde antes de publicar.", True),
        ("Palavra-chave no Título e H1", "A keyword principal deve aparecer nos primeiros 60 caracteres do título.", True),
        ("Links Internos (mínimo 3)", "Aponte para outros artigos relevantes do blog.", False),
        ("Links Externos Autoritativos", "1-2 fontes externas de qualidade por artigo.", False),
        ("Meta Description Personalizada", "Escreva manualmente.", True),
    ]

    for item, desc, ok in seo_items:
        icon = "✅" if ok else "🔧"
        cor = "#10B981" if ok else "#F59E0B"
        st.markdown(f"""
        <div style="background:var(--secondary-background-color); border:1px solid var(--faded-text-10); border-left:3px solid {cor}; border-radius:8px; padding:10px 16px; margin:6px 0; display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.1rem;">{icon}</span>
            <div>
                <strong style="color:var(--text-color);">{item}</strong>
                <p style="color:var(--text-color); font-size:0.82rem; margin:2px 0 0 0; opacity: 0.8;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Top Artigos por Views")
    if not blo_f.empty:
        top_blo = blo_plot.nlargest(10, 'Views')[['Data', 'Título', 'Tag Produto', 'Tags Tipo', 'Views', 'Tempo da Página', 'Taxa Conversão', 'Quadrante', 'Link']]
        st.dataframe(top_blo, column_config={"Data": st.column_config.DateColumn(format="DD/MM/YYYY"), "Taxa Conversão": st.column_config.NumberColumn(format="%.2f%%"), "Link": st.column_config.LinkColumn("🔗 Ler"), "Views": st.column_config.NumberColumn(format="%d")}, width="stretch", hide_index=True)

# ============================================================
# 🏷️ 5. PERFORMANCE POR TAG
# ============================================================
elif menu == "🏷️ Performance por Tag":
    st.title("Relatório por Tipo de Conteúdo")
    
    col_tags = st.columns(len(TAGS_TIPO))
    for i, (nome, keywords) in enumerate(TAGS_TIPO.items()):
        with col_tags[i]:
            st.markdown(f"""
            <div style="background:var(--secondary-background-color); border:1px solid var(--faded-text-10); border-radius:10px; padding:12px;">
                <p style="color:#0094A4; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 6px 0;">{nome}</p>
                <p style="color:var(--text-color); font-size:0.78rem; line-height:1.5; margin:0; opacity:0.8;">{', '.join(keywords[:5])}...</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### LinkedIn — Engajamento por Tipo de Conteúdo")
    if not lin_f.empty:
        from collections import defaultdict
        tag_eng = defaultdict(list)
        for _, row in lin_f.iterrows():
            for t in str(row.get('Tags Tipo', '')).split(','):
                t = t.strip()
                if t: tag_eng[t].append(row['Engajamento'])

        tag_eng_df = pd.DataFrame([{'Tag': k, 'Engajamento Médio': np.mean(v), 'Posts': len(v), 'Total': sum(v)} for k, v in tag_eng.items()]).sort_values('Engajamento Médio', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig_t1 = px.bar(tag_eng_df, x='Tag', y='Engajamento Médio', color='Engajamento Médio', color_continuous_scale='Teal')
            fig_t1.update_layout(showlegend=False)
            st.plotly_chart(fig_t1, width="stretch", theme="streamlit")
        with col2:
            fig_t2 = px.pie(tag_eng_df, names='Tag', values='Posts', hole=0.5, color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig_t2, width="stretch", theme="streamlit")

    st.markdown("#### Comparativo Cross-Channel por Produto")
    lin_prod = lin_f.groupby('Tag Produto')['Engajamento'].sum().reset_index().rename(columns={'Engajamento': 'Volume'})
    lin_prod['Canal'] = 'LinkedIn'
    blo_prod = blo_f.groupby('Tag Produto')['Views'].sum().reset_index().rename(columns={'Views': 'Volume'})
    blo_prod['Canal'] = 'Blog'
    mai_prod = mai_f.groupby('Tag Produto')['Aberturas_Abs'].sum().reset_index().rename(columns={'Aberturas_Abs': 'Volume'})
    mai_prod['Canal'] = 'E-mail'

    cross = pd.concat([lin_prod, blo_prod, mai_prod])
    if not cross.empty:
        fig_cross = px.bar(cross, x='Tag Produto', y='Volume', color='Canal', barmode='group', color_discrete_map={'LinkedIn': '#0A66C2', 'Blog': '#10B981', 'E-mail': '#F97316'})
        st.plotly_chart(fig_cross, width="stretch", theme="streamlit")

# ============================================================
# 🤖 6. IA & MODELOS PREDITIVOS
# ============================================================
elif menu == "🤖 IA & Modelos Preditivos":
    st.title("Inteligência Artificial & Modelos Preditivos")
    
    if over_f.empty:
        st.warning("⚠️ Sem dados no período selecionado para os modelos de IA.")
        st.stop()

    st.header("1. Radar de Anomalias (Z-Score Temporal)")
    evo = over_f.groupby('Data')['Tração'].sum().reset_index()
    if len(evo) > 2:
        evo['Z-Score'] = stats.zscore(evo['Tração'])
        evo['Status'] = evo['Z-Score'].apply(lambda z: '🚀 Viral' if z > 1.0 else ('🧊 Frio' if z < -1.0 else '📊 Normal'))
        CORES_Z = {'🚀 Viral': '#10B981', '📊 Normal': '#0094A4', '🧊 Frio': '#DC2626'}
        fig_z = px.bar(evo, x='Data', y='Tração', color='Status', color_discrete_map=CORES_Z, hover_data=['Z-Score'])
        st.plotly_chart(fig_z, width="stretch", theme="streamlit")
    else:
        st.info("Expanda o filtro de datas para ter mais pontos.")

    st.markdown("---")
    st.header("2. Clusterização de Campanhas (K-Means)")
    if len(mai_f) > 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        X = mai_f[['Taxa de Abertura', 'CTOR']].fillna(0)
        mai_km = mai_f.copy()
        mai_km['Cluster'] = kmeans.fit_predict(X)
        centroides = kmeans.cluster_centers_
        ranking = centroides.sum(axis=1).argsort()
        mapa = {ranking[0]: '⚠️ Baixa Atenção', ranking[1]: '📉 Média', ranking[2]: '🚀 Estrela'}
        mai_km['Classificação IA'] = mai_km['Cluster'].map(mapa)

        CORES_KM = {'🚀 Estrela': '#10B981', '📉 Média': '#F59E0B', '⚠️ Baixa Atenção': '#DC2626'}
        fig_km = px.scatter(mai_km, x='Taxa de Abertura', y='CTOR', color='Classificação IA', size='Qtd Enviados', hover_name='Título', color_discrete_map=CORES_KM)
        fig_km.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
        st.plotly_chart(fig_km, width="stretch", theme="streamlit")

    st.markdown("---")
    st.header("3. Previsibilidade de Conversão (Regressão OLS)")
    if len(blo_f) > 2:
        fig_ols = px.scatter(blo_f, x='Tempo da Página', y='Taxa Conversão', size='Views', color='Tag Produto', hover_name='Título', color_discrete_map=CORES_PRODUTOS, trendline='ols', trendline_scope='overall', trendline_color_override='#F97316')
        fig_ols.update_layout(yaxis_tickformat='.1%')
        st.plotly_chart(fig_ols, width="stretch", theme="streamlit")
    else:
        st.info("Mínimo de 3 artigos no Blog.")
