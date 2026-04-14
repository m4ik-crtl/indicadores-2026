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
# 🎨 TEMA VISUAL — AIQON B2B Dark Command Center
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
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700;
}

/* Cards informativos e Caixas Explicativas */
.info-card, .explain-box {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-left: 3px solid #0094A4;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
}
.explain-box {
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 20px;
    border-left: 1px solid rgba(128, 128, 128, 0.2);
}
.explain-box h4 {
    color: #0094A4 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.explain-box p, .info-card p {
    font-size: 0.9rem !important;
    line-height: 1.6;
    margin: 0;
}

/* Tabelas e Gráficos */
[data-testid="stDataFrame"] { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 12px; overflow: hidden; }
.js-plotly-plot { border-radius: 12px; }
hr { border-color: rgba(128, 128, 128, 0.2) !important; }

/* Customização para ocultar tags vermelhas do multiselect (se usado futuramente) */
span[data-baseweb="tag"] {
  background-color: var(--secondary-background-color) !important;
  border: 1px solid rgba(128,128,128,0.3) !important;
  color: var(--text-color) !important;
}
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
    'Institucional / Outros': '#475569',
    'Syxsense': '#A855F7'
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#8B949E', size=12),
    xaxis=dict(gridcolor='#21262D', linecolor='#30363D', tickfont=dict(color='#8B949E')),
    yaxis=dict(gridcolor='#21262D', linecolor='#30363D', tickfont=dict(color='#8B949E')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#C9D1D9')),
    margin=dict(l=10, r=10, t=30, b=10),
)

def f_br(num, is_pct=False):
    if pd.isna(num): return "0"
    if is_pct:
        return f"{num * 100:.2f}%".replace('.', ',')
    else:
        return f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def criar_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

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
    lin = pd.read_csv('dataset/linkedin_clean.csv')
    blo = pd.read_csv('dataset/blog_clean.csv')
    mai = pd.read_csv('dataset/mailchimp_clean.csv')

    lin['Data'] = pd.to_datetime(lin['Data da postagem'], errors='coerce').dt.date
    blo['Data'] = pd.to_datetime(blo['Data'], errors='coerce').dt.date
    mai['Data de Envio'] = pd.to_datetime(mai['Data de Envio'], errors='coerce').dt.date

    # ---------- MAILCHIMP ----------
    mai['Tag Produto'] = mai['Título'].apply(tag_produto)
    mai['Tags Tipo'] = mai['Título'].apply(lambda x: tags_para_str(detectar_tags(x)))
    mai['Aberturas_Abs'] = (mai['Qtd Enviados'] * mai['Taxa de Abertura']).fillna(0).round().astype(int)
    mai['Cliques_Abs'] = (mai['Qtd Enviados'] * mai['Clicks']).fillna(0).round().astype(int)

    mai_grp = mai.groupby(['Título', 'Tag Produto', 'Tags Tipo', 'url']).agg({
        'Qtd Enviados': 'sum', 'Aberturas_Abs': 'sum', 'Cliques_Abs': 'sum', 'Data de Envio': 'max'
    }).reset_index()
    mai_grp['Taxa de Abertura'] = np.where(mai_grp['Qtd Enviados'] == 0, 0,
                                            mai_grp['Aberturas_Abs'] / mai_grp['Qtd Enviados'])
    mai_grp['CTOR'] = np.where(mai_grp['Aberturas_Abs'] == 0, 0,
                                mai_grp['Cliques_Abs'] / mai_grp['Aberturas_Abs'])
    mai_grp['Ignorados'] = mai_grp['Qtd Enviados'] - mai_grp['Aberturas_Abs']
    mai_grp['Plataforma'] = 'Mailchimp'
    mai_grp['Link'] = mai_grp['url']

    # ---------- BLOG ----------
    blo['Tag Produto'] = blo['URL'].apply(tag_produto)
    blo['Tags Tipo'] = blo['URL'].apply(lambda x: tags_para_str(detectar_tags(x)))
    blo['Título'] = (blo['URL'].str.replace("https://aiqon.com.br/blog/", "", regex=False)
                     .str.replace("/", "", regex=False).str.replace("-", " ").str.title())

    blo_grp = blo.groupby(['URL', 'Título', 'Tag Produto', 'Tags Tipo']).agg({
        'Views': 'sum', 'Clicks': 'sum', 'Tempo da Página': 'mean', 'Data': 'max'
    }).reset_index()
    blo_grp['Taxa Conversão'] = np.where(blo_grp['Views'] == 0, 0,
                                          blo_grp['Clicks'] / blo_grp['Views'])
    blo_grp['Plataforma'] = 'Blog'
    blo_grp['Link'] = blo_grp['URL']

    # ---------- LINKEDIN ----------
    lin['Tag Produto'] = lin['Texto'].apply(tag_produto)
    lin['Tags Tipo'] = lin['Texto'].apply(lambda x: tags_para_str(detectar_tags(x)))
    lin['Tamanho'] = np.where(lin['Texto'].str.len() < 500, 'Curto',
                    np.where(lin['Texto'].str.len() < 1200, 'Médio', 'Longo'))
    lin['Tipo'] = np.where(
        lin['Texto'].str.contains('parceria|solução|contrat|demo|demonstr', case=False, na=False),
        'Comercial/Venda', 'Editorial/Educativo'
    )
    lin['Engajamento'] = (lin['Curtidas'].fillna(0) + lin['Comentários'].fillna(0) + lin['Shares'].fillna(0))

    lin_raw = lin.copy()
    lin_raw['Engajamento Acum'] = lin_raw.groupby('Link da Postagem')['Engajamento'].cumsum()
    lin_raw['Medicao_Nr'] = lin_raw.groupby('Link da Postagem').cumcount() + 1

    lin_grp = lin.groupby(['Link da Postagem', 'Título', 'Tag Produto', 'Tags Tipo', 'Tamanho', 'Tipo']).agg({
        'Curtidas': 'sum', 'Comentários': 'sum', 'Shares': 'sum',
        'Seguidores': 'max', 'Data': 'max', 'Engajamento': 'sum'
    }).reset_index()
    lin_grp['Taxa Engajamento (ER)'] = np.where(
        lin_grp['Seguidores'] == 0, 0,
        lin_grp['Engajamento'] / lin_grp['Seguidores']
    )
    lin_grp['Plataforma'] = 'LinkedIn'
    lin_grp['Link'] = lin_grp['Link da Postagem']

    def detectar_padrao(grupo):
        if len(grupo) < 2:
            return 'Dado Único'
        vals = grupo.sort_values('Data')['Engajamento'].values
        diffs = np.diff(vals)
        if diffs[0] > 0 and all(d <= 0 for d in diffs[1:]):
            return 'Spike Inicial 🚀'
        elif all(d >= 0 for d in diffs):
            return 'Crescimento Constante 📈'
        elif diffs[-1] < 0 and diffs[0] < 0:
            return 'Queda Rápida 📉'
        else:
            return 'Variado 〰️'

    padroes = lin_raw.groupby('Link da Postagem').apply(detectar_padrao, include_groups=False).reset_index()
    padroes.columns = ['Link da Postagem', 'Padrão']
    lin_grp = lin_grp.merge(padroes, on='Link da Postagem', how='left')
    lin_grp['Padrão'] = lin_grp['Padrão'].fillna('Dado Único')

    # ---------- OVERVIEW ----------
    r_lin = lin_grp.groupby(['Data', 'Tag Produto'])['Engajamento'].sum().reset_index().rename(columns={'Engajamento': 'Tração'})
    r_blo = blo_grp.groupby(['Data', 'Tag Produto'])['Views'].sum().reset_index().rename(columns={'Views': 'Tração'})
    r_mai = mai_grp.groupby(['Data de Envio', 'Tag Produto'])['Aberturas_Abs'].sum().reset_index().rename(
        columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Tração'})

    over = pd.concat([r_lin, r_blo, r_mai]).groupby(['Data', 'Tag Produto'])['Tração'].sum().reset_index()
    over = over.sort_values('Data')

    lista = pd.concat([
        lin_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Engajamento']].rename(
            columns={'Engajamento': 'Cliques/Tração'}),
        blo_grp[['Data', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Views']].rename(
            columns={'Views': 'Cliques/Tração'}),
        mai_grp[['Data de Envio', 'Título', 'Plataforma', 'Link', 'Tag Produto', 'Tags Tipo', 'Aberturas_Abs']].rename(
            columns={'Data de Envio': 'Data', 'Aberturas_Abs': 'Cliques/Tração'})
    ]).sort_values('Data', ascending=False)

    return over, lin_grp, lin_raw, blo_grp, mai_grp, lista


# ============================================================
# CARREGA DADOS
# ============================================================
with st.spinner("Carregando dados..."):
    df_over, df_lin, df_lin_raw, df_blo, df_mai, df_lista = carregar_dados()

# ============================================================
# 📅 DATA DA ÚLTIMA ATUALIZAÇÃO
# ============================================================
_datas_atualizacao = pd.Series(
    df_lin['Data'].dropna().tolist() +
    df_blo['Data'].dropna().tolist() +
    df_mai['Data de Envio'].dropna().tolist()
)
ULTIMA_ATUALIZACAO = _datas_atualizacao.max() if not _datas_atualizacao.empty else datetime.date.today()
ULTIMA_ATUALIZACAO_STR = pd.Timestamp(ULTIMA_ATUALIZACAO).strftime("%d/%m/%Y")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMiIgZGF0YS1uYW1lPSJMYXllciAyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODcuNDQgMTY2LjA1Ij4KICA8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiAjMDA5NGE0OwogICAgICB9CgogICAgICAuY2xzLTIgewogICAgICAgIGZpbGw6IGdyYXk7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxnIGlkPSJMYXllcl8yLTIiIGRhdGEtbmFtZT0iTGF5ZXIgMiI+CiAgICA8Zz4KICAgICAgPGc+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMzc5Ljg4LDM2LjY1Yy01LjMyLTUuMzItMTEuOTMtNy44MS0xOS40OC03Ljgxcy0xNC4wOCwyLjQ5LTE5LjQsNy45Yy01LjQ5LDUuNDktNy44MSwxMi4zNi03LjgxLDIwdjQxLjExaDEwLjgydi00MC41MWMwLTQuODksMS4zNy05LjM2LDQuNzItMTIuOTYsMy4xOC0zLjM1LDYuOTUtNC45OCwxMS41OS00Ljk4czguNDEsMS41NSwxMS41OSw0Ljk4YzMuNDMsMy42MSw0LjcyLDguMDcsNC43MiwxMi45NnY0MC41MWgxMC44MnYtNDEuMTFjLjE3LTcuNjQtMi4xNS0xNC41OS03LjY0LTIwLjA5LDAsMCwuMDksMCwuMDksMFoiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yOTEuMywyNy4xMmMtMTkuNTcsMC0zNS42MiwxNS45Ny0zNS42MiwzNS42MnMxNS45NywzNS42MiwzNS42MiwzNS42MiwzNS42Mi0xNS45NywzNS42Mi0zNS42Mi0xNS45Ny0zNS42Mi0zNS42Mi0zNS42MlpNMjkxLjMsODYuODZjLTEzLjMsMC0yNC4wMy0xMC44Mi0yNC4wMy0yNC4wM3MxMC44Mi0yNC4wMywyNC4wMy0yNC4wMywyNC4wMywxMC44MiwyNC4wMywyNC4wMy0xMC44MiwyNC4wMy0yNC4wMywyNC4wM1oiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yMTYuOCwyOC44NGMtOS4xLDAtMTYuOTEsMy4zNS0yMy4wOSw5Ljk2LTYuNTIsNi44Ny05LjI3LDE1LjE5LTkuMjcsMjQuNTVzMi44MywxNy43Nyw5LjM2LDI0LjYzYzYuMjcsNi41MiwxNC4wOCw5Ljg3LDIzLjA5LDkuODdzNi42MS0uNDMsOS43LTEuNTV2LTExLjQyYy0uMjYuMTctLjYuMjYtLjg2LjQzLTIuODMsMS4zNy01Ljg0LDEuODktOC45MywxLjg5LTYuMDEsMC0xMC43My0yLjU4LTE0Ljc2LTYuODctNC41NS00LjcyLTYuNTItMTAuNTYtNi41Mi0xNy4wOHMyLjA2LTEyLjM2LDYuNTItMTcuMDhjNC4wMy00LjI5LDguODQtNi44NywxNC43Ni02Ljg3czExLjMzLDIuMjMsMTUuNDUsNi45NWM0LjI5LDQuODksNi4xOCwxMC41Niw2LjE4LDE3djYyLjE0aDEwLjgydi02Mi4yM2MwLTkuMzYtMi44My0xNy43Ny05LjM2LTI0LjYzLTYuMDktNi41Mi0xMy45MS05Ljg3LTIzLTkuODdoMGwtLjA5LjE3aDBaIi8+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMTY1LjMsMjguODRoMTAuODJ2NjkuMDFoLTEwLjgyVjI4Ljg0WiIvPgogICAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTEyNC43LDI4Ljg0Yy05LjEsMC0xNi45MSwzLjM1LTIzLjA5LDkuOTYtNi40NCw2Ljg3LTkuMjcsMTUuMTktOS4yNywyNC41NXMyLjgzLDE3Ljc3LDkuMzYsMjQuNjNjNi4yNyw2LjUyLDE0LjA4LDkuODcsMjMuMDksOS44N3M2LjUyLS40Myw5LjYxLTEuNTV2LTExLjMzYy0uMjYuMTctLjUyLjI2LS43Ny40My0yLjgzLDEuMzctNS44NCwxLjg5LTguOTMsMS44OS02LjAxLDAtMTAuNzMtMi41OC0xNC43Ni02Ljg3LTQuNTUtNC43Mi02LjUyLTEwLjU2LTYuNTItMTcuMDhzMS45Ny0xMi4yNyw2LjUyLTE3LjA4YzQuMDMtNC4yOSw4Ljg0LTYuODcsMTQuNzYtNi44N3MxMS4zMywyLjIzLDE1LjQ1LDYuOTVjNC4yMSw0Ljg5LDYuMTgsMTAuNTYsNi4xOCwxN3YxMC44MmgwdjE0Ljg1aDB2OC43NmgxMC44MnYtMzQuNTFjMC05LjM2LTIuODMtMTcuNzctOS4zNi0yNC42My02LjE4LTYuNTItMTMuOTktOS44Ny0yMy05Ljg3aDBsLS4wOS4wOWgwWiIvPgogICAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTkuNTEsMzUuMDJjLTUuMDYsMTcuNDIsMjYuNDQsMzAuMzksMzguOCw1MS4wNywxMi4xLDIwLjM0LTEuNzIsMzEuMDctMTEuODUsMzkuNDgsNC4wMy0xMy4zLTEyLjI3LTIxLjgtMjYuMzUtMzcuNTEtMTMuOTktMTUuMjgtMTIuNzktMzYuMjItLjUyLTUzLjA1LDAsMC0uMDksMCwuMDksMFoiLz4KICAgICAgPC9nPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik00NC44NywwYy02LjE4LDIxLjM3LDM2LjkxLDMwLjIxLDM2LjY1LDYxLjgsMCw5Ljg3LTQuODEsMTguNzEtMTEuNjcsMjkuMSw1Ljg0LTExLjg1LTE0LjE2LTI2LjUyLTI4LjQxLTQyLjA2LTEzLjkxLTE1LjI4LTIxLjgtMzIuNjIsMy40My00OC44NFoiLz4KICAgIDwvZz4KICAgIDxnPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMTEuNTIsMTYwLjE0aC0yLjg3di0xNS4wOWgtNS4xNnYtMi40NmgxMy4xOXYyLjQ2aC01LjE2djE1LjA5WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMzAuNzUsMTYwLjE0aC0yLjgzdi04LjE2YzAtMS4wMi0uMjEtMS43OS0uNjItMi4yOXMtMS4wNy0uNzYtMS45Ni0uNzZjLTEuMTgsMC0yLjA1LjM1LTIuNjEsMS4wNi0uNTYuNzEtLjgzLDEuODktLjgzLDMuNTZ2Ni41OWgtMi44MnYtMTguNjhoMi44MnY0Ljc0YzAsLjc2LS4wNSwxLjU3LS4xNCwyLjQ0aC4xOGMuMzgtLjY0LjkyLTEuMTQsMS42LTEuNDkuNjgtLjM1LDEuNDgtLjUzLDIuMzktLjUzLDMuMjIsMCw0LjgyLDEuNjIsNC44Miw0Ljg2djguNjVaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTE0MC40NywxNjAuMzhjLTIuMDYsMC0zLjY4LS42LTQuODQtMS44MXMtMS43NS0yLjg2LTEuNzUtNC45Ny41NC0zLjg3LDEuNjItNS4xMSwyLjU2LTEuODYsNC40NS0xLjg2YzEuNzUsMCwzLjE0LjUzLDQuMTUsMS42LDEuMDIsMS4wNiwxLjUyLDIuNTMsMS41Miw0LjM5djEuNTJoLTguODVjLjA0LDEuMjkuMzksMi4yOCwxLjA0LDIuOTcuNjYuNjksMS41OCwxLjA0LDIuNzcsMS4wNC43OCwwLDEuNTEtLjA3LDIuMTktLjIyLjY4LS4xNSwxLjQtLjM5LDIuMTgtLjc0djIuMjljLS42OS4zMy0xLjM4LjU2LTIuMDkuN3MtMS41MS4yLTIuNDEuMlpNMTM5Ljk1LDE0OC43N2MtLjksMC0xLjYxLjI4LTIuMTYuODUtLjU0LjU3LS44NiwxLjQtLjk3LDIuNDhoNi4wM2MtLjAyLTEuMS0uMjgtMS45My0uNzktMi40OS0uNTEtLjU2LTEuMjItLjg1LTIuMTEtLjg1WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xNTUuNTIsMTQyLjZoNS4yMWMyLjQyLDAsNC4xNi4zNSw1LjI0LDEuMDZzMS42MSwxLjgyLDEuNjEsMy4zNGMwLDEuMDItLjI2LDEuODgtLjc5LDIuNTYtLjUzLjY4LTEuMjksMS4xMS0yLjI4LDEuMjh2LjEyYzEuMjMuMjMsMi4xNC42OSwyLjcyLDEuMzdjLjU4LjY4Ljg3LDEuNjEuODcsMi43OCwwLDEuNTgtLjU1LDIuODEtMS42NSwzLjctMS4xLjg5LTIuNjMsMS4zNC00LjU5LDEuMzRoLTYuMzR2LTE3LjU1Wk0xNTguMzksMTQ5Ljg1aDIuNzZjMS4yLDAsMi4wOC0uMTksMi42My0uNTdzLjgzLTEuMDMuODMtMS45NGMwLS44Mi0uMy0xLjQyLS44OS0xLjc5LS42LS4zNy0xLjU0LS41NS0yLjg0LS41NWgtMi40OXY0Ljg1Wk0xNTguMzksMTUyLjE3djUuNTZoMy4wNWMxLjIsMCwyLjExLS4yMywyLjcyLS42OXMuOTItMS4xOS45Mi0yLjE4YzAtLjkxLS4zMS0xLjU5LS45NC0yLjAzLS42Mi0uNDQtMS41Ny0uNjYtMi44NC0uNjZoLTIuOTFaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTE3OC4xOCwxNDYuNjNjLjU3LDAsMS4wNC4wNCwxLjQuMTJsLS4yOCwyLjYzYy0uNC0uMS0uODItLjE0LTEuMjUtLjE0LTEuMTMsMC0yLjA0LjM3LTIuNzQsMS4xcy0xLjA1LDEuNjktMS4wNSwyLjg3djYuOTRoLTIuODJ2LTEzLjI3aDIuMjFsLjM3LDIuMzRoLjE0Yy40NC0uNzksMS4wMS0xLjQyLDEuNzItMS44OC43MS0uNDYsMS40Ny0uNywyLjI5LS43WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xOTAuMzUsMTYwLjE0bC0uNTYtMS44NWgtLjFjLS42NC44MS0xLjI4LDEuMzYtMS45MywxLjY1cy0xLjQ4LjQ0LTIuNS40NGMtMS4zLDAtMi4zMi0uMzUtMy4wNS0xLjA2LS43My0uNy0xLjEtMS43LTEuMS0yLjk5LDAtMS4zNy41MS0yLjQsMS41Mi0zLjEsMS4wMi0uNywyLjU2LTEuMDgsNC42NC0xLjE0bDIuMjktLjA3di0uNzFjMC0uODUtLjItMS40OC0uNTktMS45LS40LS40Mi0xLjAxLS42My0xLjg0LS42My0uNjgsMC0xLjMzLjEtMS45Ni4zLS42Mi4yLTEuMjIuNDQtMS44LjcxbC0uOTEtMi4wMmMuNzItLjM4LDEuNTEtLjY2LDIuMzYtLjg2czEuNjYtLjI5LDIuNDItLjI5YzEuNjksMCwyLjk2LjM3LDMuODIsMS4xczEuMjksMS44OSwxLjI5LDMuNDd2OC45NGgtMi4wMlpNMTg2LjE1LDE1OC4yMmMxLjAyLDAsMS44NS0uMjksMi40Ny0uODZzLjkzLTEuMzguOTMtMi40MXYtMS4xNWwtMS43LjA3Yy0xLjMzLjA1LTIuMjkuMjctMi45LjY3LS42LjQtLjkxLDEtLjkxLDEuODIsMCwuNTkuMTgsMS4wNS41MywxLjM3LjM1LjMyLjg4LjQ5LDEuNTguNDlaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTIwNS4xNywxNjAuMTRoLTEwLjA3di0xLjc0bDYuNzEtOS4zN2gtNi4zdi0yLjE2aDkuNDd2MS45N2wtNi41Nyw5LjE1aDYuNzZ2Mi4xNloiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjA3Ljg5LDE0My4zNWMwLS41LjE0LS44OS40MS0xLjE3LjI4LS4yNy42Ny0uNDEsMS4xOC0uNDFzLjg4LjE0LDEuMTYuNDFjLjI4LjI3LjQxLjY2LjQxLDEuMTdzLS4xNC44Ni0uNDEsMS4xM2MtLjI4LjI4LS42Ni40MS0xLjE2LjQxcy0uOTEtLjE0LTEuMTgtLjQxYy0uMjgtLjI4LS40MS0uNjUtLjQxLTEuMTNaTTIxMC44NywxNjAuMTRoLTIuODJ2LTEzLjI3aDIuODJ2MTMuMjdaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTIxNy43MywxNjAuMTRoLTIuODJ2LTE4LjY4aDIuODJ2MTguNjhaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTIyMS41OSwxNDMuMzVjMC0uNS4xNC0uODkuNDEtMS4xNy4yOC0uMjcuNjctLjQxLDEuMTgtLjQxcy44OC4xNCwxLjE2LjQxYy4yOC4yNy40MS42Ni40MSwxLjE3cy0uMTQuODYtLjQxLDEuMTNjLS4yOC4yOC0uNjYuNDEtMS4xNi40MXMtLjkxLS4xNC0xLjE4LS40MWMtLjI4LS4yOC0uNDEtLjY1LS40MS0xLjEzWk0yMjQuNTgsMTYwLjE0aC0yLjgydi0xMy4yN2gyLjgydjEzLjI3WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0yMzYuOTIsMTYwLjE0bC0uNTYtMS44NWgtLjFjLS42NC44MS0xLjI4LDEuMzYtMS45MywxLjY1cy0xLjQ4LjQ0LTIuNS40NGMtMS4zLDAtMi4zMi0uMzUtMy4wNS0xLjA2LS43My0uNy0xLjEtMS43LTEuMS0yLjk5LDAtMS4zNy41MS0yLjQsMS41Mi0zLjEsMS4wMi0uNywyLjU2LTEuMDgsNC42NC0xLjE0bDIuMjktLjA3di0uNzFjMC0uODUtLjItMS40OC0uNTktMS45LS40LS40Mi0xLjAxLS42My0xLjg0LS42My0uNjgsMC0xLjMzLjEtMS45Ni4zLS42Mi4yLTEuMjIuNDQtMS44LjcxbC0uOTEtMi4wMmMuNzItLjM4LDEuNTEtLjY2LDIuMzYtLjg2czEuNjYtLjI5LDIuNDItLjI5YzEuNjksMCwyLjk2LjM3LDMuODIsMS4xczEuMjksMS44OSwxLjI5LDMuNDd2OC45NGgtMi4wMlpNMjMyLjcyLDE1OC4yMmMxLjAyLDAsMS44NS0uMjksMi40Ny0uODZzLjkzLTEuMzguOTMtMi40MXYtMS4xNWwtMS43LjA3Yy0xLjMzLjA1LTIuMjkuMjctMi45LjY3LS42LjQtLjkxLDEtLjkxLDEuODIsMCwuNTkuMTgsMS4wNS41MywxLjM3LjM1LjMyLjg4LjQ5LDEuNTguNDlaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTI1NC41NSwxNjAuMTRoLTIuODN2LTguMTZjMC0xLjAyLS4yMS0xLjc5LS42Mi0yLjI5cy0xLjA3LS43Ni0xLjk2LS43NmMtMS4xOSwwLTIuMDYuMzUtMi42MiwxLjA2cy0uODMsMS44OC0uODMsMy41NHY2LjYxaC0yLjgydi0xMy4yN2gyLjIxbC40LDEuNzRoLjE0Yy40LS42My45Ny0xLjEyLDEuNzEtMS40Ni43NC0uMzQsMS41NS0uNTIsMi40NS0uNTIsMy4xOCwwLDQuNzgsMS42Miw0Ljc4LDQuODZ2OC42NVoiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjcyLjYzLDE0NC44MWMtMS42NSwwLTIuOTUuNTgtMy44OSwxLjc1LS45NCwxLjE3LTEuNDIsMi43OC0xLjQyLDQuODRzLjQ1LDMuNzgsMS4zNiw0Ljg4YzAuOTEsMS4xLDIuMjIsMS42NiwzLjk0LDEuNjYuNzQsMCwxLjQ2LS4wNywyLjE2LS4yMi43LS4xNSwxLjQyLS4zNCwyLjE3LS41N3YyLjQ2Yy0xLjM4LjUyLTIuOTQuNzgtNC42OC43OC0yLjU3LDAtNC41NC0uNzgtNS45Mi0yLjMzLTEuMzgtMS41Ni0yLjA2LTMuNzgtMi4wNi02LjY4LDAtMS44Mi4zMy0zLjQyLDEtNC43OXMxLjYzLTIuNDIsMi45LTMuMTRjMS4yNi0uNzMsMi43NS0xLjA5LDQuNDUtMS4wOSwxLjc5LDAsMy40NS4zOCw0Ljk3LDEuMTNsLTEuMDMsMi4zOWMtLjU5LS4yOC0xLjIyLS41My0xLjg4LS43NC0uNjYtLjIxLTEuMzUtLjMyLTIuMDgtLjMyWiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0yNzguNDIsMTQ2Ljg3aDMuMDdsMi43LDcuNTNjLjQxLDEuMDcuNjgsMi4wOC44MiwzLjAyaC4xYy4wNy0uNDQuMi0uOTcuNC0xLjYuMTktLjYzLDEuMjEtMy42MSwzLjA1LTguOTVoMy4wNWwtNS42OCwxNS4wNGMtMS4wMywyLjc2LTIuNzUsNC4xNC01LjE2LDQuMTQtLjYyLDAtMS4yMy0uMDctMS44My0uMnYtMi4yM2MuNDIuMS45MS4xNCwxLjQ1LjE0LDEuMzYsMCwyLjMyLS43OSwyLjg3LTIuMzZsLjQ5LTEuMjUtNS4zMy0xMy4yN1oiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMzAwLjQsMTQ2LjYzYzEuNjYsMCwyLjk1LjYsMy44NywxLjguOTIsMS4yLDEuMzksMi44OCwxLjM5LDUuMDVzLS40NywzLjg3LTEuNCw1LjA4Yy0uOTQsMS4yMS0yLjI0LDEuODItMy45LDEuODJzLTIuOTktLjYtMy45MS0xLjgxaC0uMTlsLS41MiwxLjU3aC0yLjExdi0xOC42OGgyLjgydjQuNDRjMCwuMzMtLjAyLjgyLS4wNSwxLjQ2cy0uMDYsMS4wNi0uMDcsMS4yNGguMTJjLjktMS4zMiwyLjIyLTEuOTgsMy45Ni0xLjk4Wk0yOTkuNjcsMTQ4LjkzYy0xLjE0LDAtMS45NS4zMy0yLjQ1LDFzLS43NiwxLjc5LS43NywzLjM1di4xOWMwLDEuNjIuMjYsMi43OS43NywzLjUxLjUxLjcyLDEuMzUsMS4wOSwyLjUxLDEuMDksMSwwLDEuNzYtLjQsMi4yNy0xLjE5LjUyLS43OS43Ny0xLjk0Ljc3LTMuNDMsMC0zLjAyLTEuMDMtNC41Mi0zLjEtNC41MloiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMzE0LjczLDE2MC4zOGMtMi4wNiwwLTMuNjgtLjYtNC44NC0xLjgxcy0xLjc1LTIuODYtMS43NS00Ljk3LjU0LTMuODcsMS42Mi01LjExLDIuNTYtMS44Niw0LjQ1LTEuODZjMS43NSwwLDMuMTQuNTMsNC4xNSwxLjYsMS4wMiwxLjA2LDEuNTIsMi41MywxLjUyLDQuMzl2MS41MmgtOC44NWMuMDQsMS4yOS4zOSwyLjI4LDEuMDQsMi45Ny42Ni42OSwxLjU4LDEuMDQsMi43NywxLjA0Ljc4LDAsMS41MS0uMDcsMi4xOS0uMjIuNjgtLjE1LDEuNC0uMzksMi4xOC0uNzR2Mi4yOWMtLjY5LjMzLTEuMzguNTYtMi4wOS43cy0xLjUxLjItMi40MS4yWk0zMTQuMjIsMTQ4Ljc3Yy0uOSwwLTEuNjEuMjgtMi4xNi44NS0uNTQuNTctLjg2LDEuNC0uOTcsMi40OGg2LjAzYy0uMDItMS4xLS4yOC0xLjkzLS43OS0yLjQ5LS41MS0uNTYtMS4yMi0uODUtMi4xMS0uODVaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTMyOS44MywxNDYuNjNjLjU3LDAsMS4wNC4wNCwxLjQuMTJsLS4yOCwyLjYzYy0uNC0uMS0uODItLjE0LTEuMjUtLjE0LTEuMTMsMC0yLjA0LjM3LTIuNzQsMS4xcy0xLjA1LDEuNjktMS4wNSwyLjg3djYuOTRoLTIuODJ2LTEzLjI3aDIuMjFsLjM3LDIuMzRoLjE0Yy40NC0uNzksMS4wMS0xLjQyLDEuNzItMS44OC43MS0uNDYsMS40Ny0uNywyLjI5LS43WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0zNTQuMjIsMTYwLjE0aC0yLjg4di03LjkxaC04LjA5djcuOTFoLTIuODd2LTE3LjU1aDIuODd2Ny4xOGg4LjA5di03LjE4aDIuODh2MTcuNTVaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTM2Ny45LDE2MC4xNGwtLjQtMS43NGgtLjE0Yy0uMzkuNjItLjk1LDEuMS0xLjY3LDEuNDUtLjcyLjM1LTEuNTUuNTMtMi40OC41M2MtMS42MSwwLTIuODEtLjQtMy42LTEuMi0uNzktLjgtMS4xOS0yLjAxLTEuMTktMy42NHYtOC42OGgyLjg0djguMTljMCwxLjAyLjIxLDEuNzguNjIsMi4yOS40Mi41MSwxLjA3Ljc2LDEuOTYuNzYsMS4xOCwwLDIuMDUtLjM1LDIuNjEtMS4wNi41Ni0uNzEuODMtMS44OS44My0zLjU2di02LjYxaDIuODN2MTMuMjdoLTIuMjJaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTM4MC45MiwxNDYuNjNjMS42NiwwLDIuOTUuNiwzLjg3LDEuOC45MiwxLjIsMS4zOSwyLjg4LDEuMzksNS4wNXMtLjQ3LDMuODctMS40LDUuMDhjLS45NCwxLjIxLTIuMjQsMS44Mi0zLjksMS44MnMtMi45OS0uNi0zLjkxLTEuODFoLS4xOWwtLjUyLDEuNTdoLTIuMTF2LTE4LjY4aDIuODJ2NC40NGMwLC4zMy0uMDIuODItLjA1LDEuNDZzLS4wNiwxLjA2LS4wNywxLjI0aC4xMmMuOS0xLjMyLDIuMjItMS45OCwzLjk2LTEuOThaTTM4MC4xOSwxNDguOTNjLTEuMTQsMC0xLjk1LjMzLTIuNDUsMXMtLjc2LDEuNzktLjc3LDMuMzV2LjE5YzAsMS42Mi4yNiwyLjc5Ljc3LDMuNTEuNTEuNzIsMS4zNSwxLjA5LDIuNTEsMS4wOSwxLDAsMS43Ni0uNCwyLjI3LTEuMTkuNTItLjc5Ljc3LTEuOTQuNzctMy40MywwLTMuMDItMS4wMy00LjUyLTMuMS00LjUyWiIvPgogICAgPC9nPgogIDwvZz4KPC9zdmc+", width=180)
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
filtro_prod = []
with st.sidebar.expander("📌 Selecionar Produtos", expanded=False):
    for p in prod_disp:
        if st.checkbox(p, value=True, key=f"prod_{p}"):
            filtro_prod.append(p)

tags_disp = ['Produto', 'Notícia', 'Conceito', 'Campanha / Ebook', 'Comercial', 'Editorial / Educativo']
filtro_tag = []
with st.sidebar.expander("🏷️ Tipo de Conteúdo", expanded=False):
    for t in tags_disp:
        if st.checkbox(t, value=True, key=f"tag_{t}"):
            filtro_tag.append(t)

if len(datas) == 2 and filtro_prod and filtro_tag:
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
# 📅 BADGE DE ATUALIZAÇÃO NA SIDEBAR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="
    background: rgba(0, 148, 164, 0.12);
    border: 1px solid rgba(0, 148, 164, 0.35);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
">
    <p style="
        color: #0094A4;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 0 0 4px 0;
    ">Última Atualização</p>
    <p style="
        color: #C9D1D9;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        font-family: 'DM Mono', monospace;
    ">{ULTIMA_ATUALIZACAO_STR}</p>
    <p style="
        color: #7D8590;
        font-size: 0.72rem;
        margin: 4px 0 0 0;
    ">Dados via Colab → GitHub → Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 📘 HELPERS
# ============================================================
def explain(titulo, descricao):
    st.markdown(f"""
    <div class="explain-box">
        <h4>{titulo}</h4>
        <p>{descricao}</p>
    </div>
    """, unsafe_allow_html=True)

def badge_atualizacao():
    st.markdown(f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 148, 164, 0.1);
        border: 1px solid rgba(0, 148, 164, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        margin-bottom: 12px;
    ">
        <span style="
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #0094A4;
            display: inline-block;
        "></span>
        <span style="color: #7D8590; font-size: 0.78rem;">Dados atualizados em</span>
        <span style="
            color: #0094A4;
            font-size: 0.78rem;
            font-weight: 600;
            font-family: 'DM Mono', monospace;
        ">{ULTIMA_ATUALIZACAO_STR}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 🌐 1. OVERVIEW GERAL
# ============================================================
if menu == "🌐 Overview Geral":
    st.title("Overview de Marketing")
    badge_atualizacao()
    explain("O que é este painel?",
            "Visão consolidada de toda a presença digital: LinkedIn, Blog e E-mail. "
            "O indicador <strong>Tração</strong> soma engajamentos do LinkedIn, visualizações do Blog e aberturas de e-mail — "
            "representando o total de 'toques' da marca no período selecionado.")

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
        explain("Como ler este gráfico",
                "Cada barra representa o volume total de interações num dia. "
                "A linha tracejada é a média do período — dias acima dela foram acima do esperado.")

        evo = over_f.groupby('Data')['Tração'].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=evo['Data'], y=evo['Tração'],
                             marker_color='#0094A4', name='Tração Diária',
                             hovertemplate='<b>%{x}</b><br>Tração: %{y:,.0f}<extra></extra>'))
        if not evo.empty and len(evo) > 1:
            media = evo['Tração'].mean()
            fig.add_hline(y=media, line_dash='dash', line_color='#F97316',
                          annotation_text=f'Média: {f_br(media)}',
                          annotation_font_color='#F97316')
        criar_fig(fig)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("#### Tração por Produto")
        explain("Como ler",
                "Mostra quais produtos geraram mais atenção no período. "
                "Use para decidir onde intensificar ou reduzir esforços.")
        rank = over_f.groupby('Tag Produto')['Tração'].sum().reset_index().sort_values('Tração', ascending=True)
        rank = rank[rank['Tag Produto'] != 'Institucional / Outros']
        if not rank.empty:
            fig2 = px.bar(rank, y='Tag Produto', x='Tração', orientation='h',
                          color='Tag Produto', color_discrete_map=CORES_PRODUTOS)
            fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Distribuição por Canal")
    explain("Proporção de canais",
            "Entenda quanto cada canal contribui para a tração total. "
            "Um canal dominante pode indicar dependência — considere equilibrar os investimentos.")
    lin_total = lin_f['Engajamento'].sum() if not lin_f.empty else 0
    blo_total = blo_f['Views'].sum() if not blo_f.empty else 0
    mai_total = mai_f['Aberturas_Abs'].sum() if not mai_f.empty else 0
    canais = pd.DataFrame({
        'Canal': ['LinkedIn', 'Blog', 'E-mail'],
        'Volume': [lin_total, blo_total, mai_total],
        'Cor': ['#0A66C2', '#10B981', '#F97316']
    })
    if canais['Volume'].sum() > 0:
        fig_pie = go.Figure(go.Pie(
            labels=canais['Canal'], values=canais['Volume'],
            hole=0.6,
            marker=dict(colors=['#0A66C2', '#10B981', '#F97316']),
            textinfo='label+percent',
            textfont=dict(color='#C9D1D9', size=13)
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=280)
        fig_pie.update_traces(hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<extra></extra>')
        st.plotly_chart(fig_pie, width="stretch")

    st.markdown("#### Lista Mestra de Conteúdo")
    st.dataframe(
        lista_f[['Data', 'Plataforma', 'Título', 'Tag Produto', 'Tags Tipo', 'Cliques/Tração', 'Link']],
        column_config={
            "Link": st.column_config.LinkColumn("🔗 Abrir"),
            "Cliques/Tração": st.column_config.NumberColumn(format="%d"),
        },
        width="stretch", hide_index=True
    )


# ============================================================
# 💼 2. LINKEDIN
# ============================================================
elif menu == "💼 LinkedIn — Engajamento":
    st.title("LinkedIn — Análise de Engajamento")
    badge_atualizacao()
    explain("O que analisamos aqui?",
            "Engajamento = Curtidas + Comentários + Shares. "
            "O <strong>ER (Engagement Rate)</strong> divide o engajamento pelo número de seguidores — permite comparar posts "
            "mesmo quando o número de seguidores muda ao longo do tempo. "
            "Conteúdos com ER acima de 0,5% são considerados fortes para B2B.")

    c1, c2, c3, c4 = st.columns(4)
    media_eng = lin_f['Engajamento'].mean() if not lin_f.empty else 0
    er_global = (lin_f['Engajamento'].sum() / lin_f['Seguidores'].sum()
                 if not lin_f.empty and lin_f['Seguidores'].sum() > 0 else 0)
    n_posts = len(lin_f)
    top_post = lin_f['Engajamento'].max() if not lin_f.empty else 0

    c1.metric("Média Engajamento/Post", f_br(media_eng))
    c2.metric("ER Global", f_br(er_global, True), help="Engajamento Total ÷ Seguidores Máx")
    c3.metric("Posts Analisados", str(n_posts))
    c4.metric("Maior Engajamento", f_br(top_post))

    st.markdown("---")
    st.markdown("#### Curva de Engajamento ao Longo do Tempo")
    explain("Por que isso importa?",
            "Posts de <strong>notícia</strong> tendem a ter spike rápido e queda. "
            "Posts de <strong>conceito</strong> crescem mais devagar mas de forma constante. "
            "Identificar esses padrões permite ajustar cadência e tipo de conteúdo. "
            "Cada linha representa um post — passe o mouse para ver o título.")

    posts_multi = lin_raw_f.groupby('Link da Postagem').filter(lambda x: len(x) > 1)
    if not posts_multi.empty:
        top_posts = (lin_f.nlargest(10, 'Engajamento')['Link da Postagem'].tolist())
        curva_df = posts_multi[posts_multi['Link da Postagem'].isin(top_posts)]
        fig_curva = px.line(
            curva_df.sort_values('Data'),
            x='Data', y='Engajamento', color='Título',
            markers=True, line_shape='spline',
            title='Top 10 Posts — Engajamento por Medição'
        )
        fig_curva.update_layout(**PLOTLY_LAYOUT)
        fig_curva.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Data: %{x}<br>Engajamento: %{y}<extra></extra>')
        st.plotly_chart(fig_curva, width="stretch")
    else:
        st.markdown("""
        <div class="info-card">
        <p>⏳ <strong>Dados de medição única detectados.</strong> Para ver as curvas de engajamento (24h → 48h → 72h → 7 dias),
        registre o mesmo post na planilha em múltiplas datas após a publicação.
        Com os dados atuais, veja abaixo a evolução temporal agregada por semana.</p>
        </div>
        """, unsafe_allow_html=True)

        lin_temporal = lin_f.copy()
        lin_temporal['Semana'] = pd.to_datetime(lin_temporal['Data']).dt.to_period('W').astype(str)
        semanal = lin_temporal.groupby('Semana')['Engajamento'].mean().reset_index()
        fig_sem = px.line(semanal, x='Semana', y='Engajamento', markers=True,
                          title='Engajamento Médio por Semana')
        fig_sem.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_sem, width="stretch")

    st.markdown("#### Padrões de Comportamento dos Posts")
    explain("Como classificamos os padrões",
            "<strong>Spike Inicial 🚀</strong>: pico no 1º registro, queda depois — típico de notícias urgentes. "
            "<strong>Crescimento Constante 📈</strong>: engajamento aumenta progressivamente — conteúdos evergreen ou com boa distribuição orgânica. "
            "<strong>Queda Rápida 📉</strong>: já começa baixo — pode indicar horário ruim ou tema irrelevante. "
            "<strong>Variado 〰️</strong>: comportamento misto.")
    if 'Padrão' in lin_f.columns:
        pad_count = lin_f['Padrão'].value_counts().reset_index()
        pad_count.columns = ['Padrão', 'Posts']
        fig_pad = px.bar(pad_count, x='Padrão', y='Posts',
                         color='Padrão',
                         color_discrete_sequence=['#10B981', '#0094A4', '#DC2626', '#F59E0B'])
        fig_pad.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_pad, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Engajamento por Tipo × Tamanho")
        explain("Como ler",
                "Cruza <strong>Tipo</strong> (Editorial vs Comercial) com <strong>Tamanho</strong> do post. "
                "Células mais escuras = maior engajamento médio. Use para calibrar o formato ideal.")
        ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
        if not ab.empty:
            fig_hm = px.density_heatmap(ab, x='Tipo', y='Tamanho', z='Engajamento',
                                         text_auto='.1f', color_continuous_scale='Teal')
            fig_hm.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_hm, width="stretch")

    with col2:
        st.markdown("#### Top Tags de Conteúdo")
        explain("Como ler",
                "Distribuição dos posts por tipo de conteúdo identificado automaticamente. "
                "Um bom mix B2B tem ~40% Produto, ~30% Conceito/Educativo e ~30% Comercial/Notícia.")
        from collections import Counter
        todas_tags = []
        for row in lin_f['Tags Tipo'].dropna():
            todas_tags.extend([t.strip() for t in str(row).split(',')])
        if todas_tags:
            tag_df = pd.DataFrame(Counter(todas_tags).most_common(), columns=['Tag', 'Posts'])
            fig_tags = px.bar(tag_df, x='Posts', y='Tag', orientation='h',
                              color_discrete_sequence=['#0094A4'])
            fig_tags.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_tags, width="stretch")

    st.markdown("#### Todos os Posts")
    cols_show = ['Data', 'Título', 'Tag Produto', 'Tags Tipo', 'Tipo', 'Tamanho', 'Engajamento', 'Taxa Engajamento (ER)', 'Link']
    cols_show = [c for c in cols_show if c in lin_f.columns]
    st.dataframe(
        lin_f[cols_show].sort_values('Engajamento', ascending=False),
        column_config={
            "Taxa Engajamento (ER)": st.column_config.NumberColumn(format="%.4f"),
            "Link": st.column_config.LinkColumn("🔗 Ver Post"),
            "Engajamento": st.column_config.NumberColumn(format="%d"),
        },
        width="stretch", hide_index=True
    )


# ============================================================
# 📧 3. E-MAIL MARKETING
# ============================================================
elif menu == "📧 E-mail Marketing":
    st.title("E-mail Marketing — Funil e Conversão")
    badge_atualizacao()
    explain("Métricas principais",
            "<strong>Taxa de Abertura</strong>: % dos destinatários que abriram o e-mail. Benchmarks B2B: >25% bom, >35% excelente. "
            "<strong>CTOR (Click-to-Open Rate)</strong>: % dos que abriram e clicaram. Mede a qualidade do conteúdo do e-mail. "
            "Benchmarks B2B: >10% bom, >20% excelente. "
            "O <strong>Quadrante</strong> abaixo permite classificar cada campanha em 4 categorias.")

    ctor_g = (mai_f['Cliques_Abs'].sum() / mai_f['Aberturas_Abs'].sum()
              if not mai_f.empty and mai_f['Aberturas_Abs'].sum() > 0 else 0)
    tx_ab_g = (mai_f['Aberturas_Abs'].sum() / mai_f['Qtd Enviados'].sum()
               if not mai_f.empty and mai_f['Qtd Enviados'].sum() > 0 else 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taxa de Abertura", f_br(tx_ab_g, True), help="Aberturas ÷ Total Enviados")
    c2.metric("CTOR Global", f_br(ctor_g, True), help="Cliques ÷ Aberturas")
    c3.metric("Total Enviados", f_br(mai_f['Qtd Enviados'].sum()))
    c4.metric("Campanhas", str(len(mai_f)))

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Funil de E-mail")
        explain("Funil",
                "Cada etapa representa a conversão para a próxima ação. "
                "Um funil saudável B2B tem boa proporção de aberturas (assunto atraente) "
                "e CTOR (conteúdo relevante).")
        fig_funil = go.Figure(go.Funnel(
            y=['Enviados', 'Aberturas', 'Cliques'],
            x=[mai_f['Qtd Enviados'].sum(), mai_f['Aberturas_Abs'].sum(), mai_f['Cliques_Abs'].sum()],
            textinfo='value+percent initial',
            marker={'color': ['#1E293B', '#0094A4', '#10B981']},
            connector={'fillcolor': '#161B22'}
        ))
        fig_funil.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_funil, width="stretch")

    with col2:
        st.markdown("#### Quadrante de Desempenho: Abertura × CTOR")
        explain("Como usar o quadrante",
                "Eixo X = Taxa de Abertura (qualidade do assunto/remetente). "
                "Eixo Y = CTOR (qualidade do conteúdo dentro do e-mail). "
                "🌟 <strong>Estrelas</strong>: alto em ambos. "
                "📣 <strong>Bom Assunto</strong>: muito aberto mas pouco clicado — conteúdo interno fraco. "
                "💎 <strong>Nicho Quente</strong>: poucos abrem mas quem abre, clica muito. "
                "⚠️ <strong>Revisar</strong>: baixo em ambos.")
        if not mai_f.empty:
            mai_plot = mai_f.copy()
            mai_plot['Diff CTOR'] = np.where(
                ctor_g == 0, 0,
                (mai_plot['CTOR'] - ctor_g) / (ctor_g + 1e-9)
            )

            def quadrante(row):
                acima_ab = row['Taxa de Abertura'] >= tx_ab_g
                acima_ctor = row['CTOR'] >= ctor_g
                if acima_ab and acima_ctor: return '🌟 Estrela'
                elif acima_ab and not acima_ctor: return '📣 Bom Assunto'
                elif not acima_ab and acima_ctor: return '💎 Nicho Quente'
                else: return '⚠️ Revisar'

            mai_plot['Quadrante'] = mai_plot.apply(quadrante, axis=1)

            CORES_Q = {
                '🌟 Estrela': '#10B981',
                '📣 Bom Assunto': '#F59E0B',
                '💎 Nicho Quente': '#0094A4',
                '⚠️ Revisar': '#DC2626'
            }

            fig_scat = px.scatter(
                mai_plot, x='Taxa de Abertura', y='CTOR',
                size='Qtd Enviados', color='Quadrante',
                hover_name='Título',
                color_discrete_map=CORES_Q,
                custom_data=['Aberturas_Abs', 'Cliques_Abs', 'Qtd Enviados']
            )
            fig_scat.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Abertura: %{x:.1%}<br>CTOR: %{y:.1%}<br>"
                    "Enviados: %{customdata[2]:,.0f}<br>"
                    "Aberturas: %{customdata[0]:,.0f} | Cliques: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                )
            )
            fig_scat.add_vline(x=tx_ab_g, line_dash='dot', line_color='#475569',
                               annotation_text='Média Abertura', annotation_font_color='#475569')
            fig_scat.add_hline(y=ctor_g, line_dash='dot', line_color='#475569',
                               annotation_text='Média CTOR', annotation_font_color='#475569')
            fig_scat.update_layout(**PLOTLY_LAYOUT,
                                   xaxis_tickformat='.1%', yaxis_tickformat='.1%', height=360)
            st.plotly_chart(fig_scat, width="stretch")

    st.markdown("#### Evolução da Taxa de Abertura ao Longo do Tempo")
    explain("Tendência temporal",
            "Monitore se as taxas de abertura estão subindo ou caindo ao longo do tempo. "
            "Uma queda consistente pode indicar lista desatualizada ou fadiga de conteúdo.")
    if not mai_f.empty:
        mai_evo = mai_f.sort_values('Data de Envio')
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(
            x=mai_evo['Data de Envio'], y=mai_evo['Taxa de Abertura'],
            mode='lines+markers', name='Abertura',
            line=dict(color='#0094A4', width=2),
            marker=dict(size=7),
            hovertemplate='<b>%{text}</b><br>Abertura: %{y:.1%}<extra></extra>',
            text=mai_evo['Título']
        ))
        fig_evo.add_trace(go.Scatter(
            x=mai_evo['Data de Envio'], y=mai_evo['CTOR'],
            mode='lines+markers', name='CTOR',
            line=dict(color='#F97316', width=2, dash='dash'),
            marker=dict(size=7),
            hovertemplate='<b>%{text}</b><br>CTOR: %{y:.1%}<extra></extra>',
            text=mai_evo['Título']
        ))
        fig_evo.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat='.1%', height=300)
        st.plotly_chart(fig_evo, width="stretch")

    st.markdown("#### Todas as Campanhas")
    if not mai_f.empty and 'Quadrante' in mai_plot.columns:
        tabela_mai = mai_f.merge(mai_plot[['Título', 'Quadrante']], on='Título', how='left')
        st.dataframe(
            tabela_mai[['Data de Envio', 'Título', 'Tag Produto', 'Tags Tipo', 'Qtd Enviados',
                        'Aberturas_Abs', 'Cliques_Abs', 'Taxa de Abertura', 'CTOR', 'Quadrante']].sort_values('CTOR', ascending=False),
            column_config={
                "Taxa de Abertura": st.column_config.NumberColumn(format="%.2f%%"),
                "CTOR": st.column_config.NumberColumn(format="%.2f%%"),
                "Aberturas_Abs": st.column_config.NumberColumn("Aberturas", format="%d"),
                "Cliques_Abs": st.column_config.NumberColumn("Cliques", format="%d"),
            },
            width="stretch", hide_index=True
        )


# ============================================================
# 📝 4. BLOG & SEO
# ============================================================
elif menu == "📝 Blog & SEO":
    st.title("Blog — Retenção, Conversão e SEO")
    badge_atualizacao()
    explain("Métricas do Blog",
            "<strong>Views</strong>: número total de visualizações do artigo. "
            "<strong>Tempo na Página</strong>: tempo médio de leitura — sinal de qualidade do conteúdo. Bom para B2B: >3 min (180s). "
            "<strong>Taxa de Conversão</strong>: % dos leitores que clicaram em algum CTA/link interno. "
            "O gráfico de dispersão abaixo cruza retenção e conversão — o post ideal fica no canto superior direito.")

    tempo_g = blo_f['Tempo da Página'].mean() if not blo_f.empty else 0
    tx_conv_g = (blo_f['Clicks'].sum() / blo_f['Views'].sum()
                 if not blo_f.empty and blo_f['Views'].sum() > 0 else 0)
    total_views = blo_f['Views'].sum() if not blo_f.empty else 0
    n_artigos = len(blo_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tempo Médio na Página", f"{f_br(tempo_g)}s", help="Quanto tempo em média os leitores ficam no artigo")
    c2.metric("Taxa de Conversão Média", f_br(tx_conv_g, True), help="Cliques em CTAs ÷ Views totais")
    c3.metric("Visualizações Totais", f_br(total_views))
    c4.metric("Artigos Monitorados", str(n_artigos))

    st.markdown("#### Quadrante: Retenção × Conversão")
    explain("4 categorias de artigos",
            "🌟 <strong>Magnetizadores</strong>: longo tempo + alta conversão — os melhores artigos. "
            "📖 <strong>Leitura Profunda</strong>: muito lidos mas sem conversão — falta CTA. "
            "⚡ <strong>Conversores Rápidos</strong>: convertem mesmo com pouco tempo — assunto urgente ou CTA agressivo. "
            "🔧 <strong>Revisar</strong>: baixo em ambos — reformular conteúdo ou distribuição.")
    if not blo_f.empty:
        blo_plot = blo_f.copy()

        def q_blog(row):
            acima_t = row['Tempo da Página'] >= tempo_g
            acima_c = row['Taxa Conversão'] >= tx_conv_g
            if acima_t and acima_c: return '🌟 Magnetizador'
            elif acima_t and not acima_c: return '📖 Leitura Profunda'
            elif not acima_t and acima_c: return '⚡ Conversor Rápido'
            else: return '🔧 Revisar'

        blo_plot['Quadrante'] = blo_plot.apply(q_blog, axis=1)

        CORES_QB = {
            '🌟 Magnetizador': '#10B981',
            '📖 Leitura Profunda': '#0094A4',
            '⚡ Conversor Rápido': '#F59E0B',
            '🔧 Revisar': '#DC2626'
        }

        fig_blo = px.scatter(
            blo_plot, x='Tempo da Página', y='Taxa Conversão',
            size='Views', color='Quadrante', hover_name='Título',
            color_discrete_map=CORES_QB,
            custom_data=['Views', 'Tags Tipo']
        )
        fig_blo.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Tempo: %{x:.0f}s<br>Conversão: %{y:.1%}<br>"
                "Views: %{customdata[0]:,.0f}<br>Tag: %{customdata[1]}"
                "<extra></extra>"
            )
        )
        fig_blo.add_vline(x=tempo_g, line_dash='dot', line_color='#475569',
                          annotation_text='Média Tempo', annotation_font_color='#475569')
        fig_blo.add_hline(y=tx_conv_g, line_dash='dot', line_color='#475569',
                          annotation_text='Média Conv.', annotation_font_color='#475569')
        fig_blo.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat='.1%', height=420)
        st.plotly_chart(fig_blo, width="stretch")

    st.markdown("---")
    st.markdown("#### Checklist de SEO e Qualidade de Conteúdo")
    explain("Por que isso importa?",
            "Artigos com SEO score verde no plugin (Yoast/Rank Math) e bons links internos tendem a "
            "ranquear melhor e ter mais tráfego orgânico. Conteúdos gerados por IA precisam de atenção especial "
            "pois costumam ter links internos/externos pobres.")

    seo_items = [
        ("SEO Score Verde no Plugin", "Todo artigo deve atingir score verde (Yoast/Rank Math) antes de publicar.", True),
        ("Palavra-chave no Título e H1", "A keyword principal deve aparecer nos primeiros 60 caracteres do título.", True),
        ("Links Internos (mínimo 3)", "Aponte para outros artigos relevantes do blog — melhora o tempo na página.", False),
        ("Links Externos Autoritativos", "1-2 fontes externas de qualidade (CVE DB, fabricantes, pesquisas) por artigo.", False),
        ("Meta Description Personalizada", "Não deixe o plugin gerar automaticamente — escreva manualmente.", True),
        ("Conteúdo IA Revisado", "IA gera texto com links pobres. Sempre adicionar links internos/externos manualmente.", False),
        ("CTA Interno Claro", "Cada artigo deve ter pelo menos 1 CTA (ex: link para produto ou formulário de contato).", False),
        ("Imagem com Alt Text", "Todas as imagens precisam ter texto alternativo descritivo.", True),
    ]

    for item, desc, ok in seo_items:
        icon = "✅" if ok else "🔧"
        cor = "#10B981" if ok else "#F59E0B"
        st.markdown(f"""
        <div style="background:#161B22; border:1px solid #21262D; border-left:3px solid {cor};
                    border-radius:8px; padding:10px 16px; margin:6px 0; display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.1rem;">{icon}</span>
            <div>
                <strong style="color:#C9D1D9;">{item}</strong>
                <p style="color:#7D8590; font-size:0.82rem; margin:2px 0 0 0;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Top Artigos por Views")
    if not blo_f.empty:
        top_blo = blo_plot.nlargest(10, 'Views')[
            ['Data', 'Título', 'Tag Produto', 'Tags Tipo', 'Views',
             'Tempo da Página', 'Taxa Conversão', 'Quadrante', 'Link']
        ]
        st.dataframe(
            top_blo,
            column_config={
                "Taxa Conversão": st.column_config.NumberColumn(format="%.2f%%"),
                "Link": st.column_config.LinkColumn("🔗 Ler"),
                "Views": st.column_config.NumberColumn(format="%d"),
            },
            width="stretch", hide_index=True
        )


# ============================================================
# 🏷️ 5. PERFORMANCE POR TAG
# ============================================================
elif menu == "🏷️ Performance por Tag":
    st.title("Relatório por Tipo de Conteúdo")
    badge_atualizacao()
    explain("Como usar este relatório",
            "Compare a performance entre diferentes <strong>tipos de conteúdo</strong> (Produto, Notícia, Conceito, Campanha). "
            "Isso permite entender qual abordagem funciona melhor para cada canal e ajustar a estratégia editorial. "
            "Use os filtros da sidebar para refinar a análise.")

    st.markdown("#### Sistema de Tags — Como Classificamos o Conteúdo")
    col_tags = st.columns(len(TAGS_TIPO))
    for i, (nome, keywords) in enumerate(TAGS_TIPO.items()):
        with col_tags[i]:
            st.markdown(f"""
            <div style="background:#161B22; border:1px solid #21262D; border-radius:10px; padding:12px;">
                <p style="color:#0094A4; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 6px 0;">{nome}</p>
                <p style="color:#7D8590; font-size:0.78rem; line-height:1.5; margin:0;">{', '.join(keywords[:5])}...</p>
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
                if t:
                    tag_eng[t].append(row['Engajamento'])

        tag_eng_df = pd.DataFrame([
            {'Tag': k, 'Engajamento Médio': np.mean(v), 'Posts': len(v), 'Total': sum(v)}
            for k, v in tag_eng.items()
        ]).sort_values('Engajamento Médio', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            explain("Engajamento médio", "Qual tipo de post gera mais engajamento médio por publicação.")
            fig_t1 = px.bar(tag_eng_df, x='Tag', y='Engajamento Médio',
                            color='Engajamento Médio', color_continuous_scale='Teal')
            fig_t1.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_t1, width="stretch")
        with col2:
            explain("Volume de posts", "Quanto de cada tipo foi produzido — mostra o mix editorial atual.")
            fig_t2 = px.pie(tag_eng_df, names='Tag', values='Posts', hole=0.5,
                            color_discrete_sequence=px.colors.sequential.Teal)
            fig_t2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_t2, width="stretch")

    st.markdown("#### E-mail — Abertura Média por Tipo")
    if not mai_f.empty:
        from collections import defaultdict
        tag_mai = defaultdict(list)
        for _, row in mai_f.iterrows():
            for t in str(row.get('Tags Tipo', '')).split(','):
                t = t.strip()
                if t:
                    tag_mai[t].append(row['Taxa de Abertura'])

        tag_mai_df = pd.DataFrame([
            {'Tag': k, 'Abertura Média': np.mean(v), 'Campanhas': len(v)}
            for k, v in tag_mai.items()
        ]).sort_values('Abertura Média', ascending=False)

        fig_mai_tag = px.bar(tag_mai_df, x='Tag', y='Abertura Média',
                             color='Abertura Média',
                             color_continuous_scale='Teal',
                             text=tag_mai_df['Abertura Média'].map('{:.1%}'.format))
        fig_mai_tag.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat='.1%', showlegend=False)
        st.plotly_chart(fig_mai_tag, width="stretch")

    st.markdown("#### Blog — Views Médias por Tipo")
    if not blo_f.empty:
        from collections import defaultdict
        tag_blo = defaultdict(list)
        for _, row in blo_f.iterrows():
            for t in str(row.get('Tags Tipo', '')).split(','):
                t = t.strip()
                if t:
                    tag_blo[t].append(row['Views'])

        tag_blo_df = pd.DataFrame([
            {'Tag': k, 'Views Médias': np.mean(v), 'Artigos': len(v)}
            for k, v in tag_blo.items()
        ]).sort_values('Views Médias', ascending=False)

        fig_blo_tag = px.bar(tag_blo_df, x='Tag', y='Views Médias',
                             color='Views Médias', color_continuous_scale='Teal',
                             text=tag_blo_df['Views Médias'].map('{:.0f}'.format))
        fig_blo_tag.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_blo_tag, width="stretch")

    st.markdown("#### Comparativo Cross-Channel por Produto")
    explain("Canal mais eficiente por produto",
            "Cruza produto com canal para identificar onde cada marca performa melhor. "
            "Exemplo: um produto técnico pode ter melhor resultado no blog, enquanto outro converte melhor via e-mail.")

    lin_prod = lin_f.groupby('Tag Produto')['Engajamento'].sum().reset_index().rename(columns={'Engajamento': 'Volume'})
    lin_prod['Canal'] = 'LinkedIn'
    blo_prod = blo_f.groupby('Tag Produto')['Views'].sum().reset_index().rename(columns={'Views': 'Volume'})
    blo_prod['Canal'] = 'Blog'
    mai_prod = mai_f.groupby('Tag Produto')['Aberturas_Abs'].sum().reset_index().rename(columns={'Aberturas_Abs': 'Volume'})
    mai_prod['Canal'] = 'E-mail'

    cross = pd.concat([lin_prod, blo_prod, mai_prod])
    if not cross.empty:
        fig_cross = px.bar(cross, x='Tag Produto', y='Volume', color='Canal', barmode='group',
                           color_discrete_map={'LinkedIn': '#0A66C2', 'Blog': '#10B981', 'E-mail': '#F97316'})
        fig_cross.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_cross, width="stretch")


# ============================================================
# 🤖 6. IA & MODELOS PREDITIVOS
# ============================================================
elif menu == "🤖 IA & Modelos Preditivos":
    st.title("Inteligência Artificial & Modelos Preditivos")
    badge_atualizacao()
    explain("O que a IA faz aqui?",
            "Três algoritmos analisam seus dados de marketing automaticamente: "
            "(1) <strong>Detecção de Anomalias</strong> identifica dias virais e dias frios usando Z-Score estatístico. "
            "(2) <strong>K-Means</strong> agrupa suas campanhas de e-mail por padrão de performance sem supervisão humana. "
            "(3) <strong>Regressão OLS</strong> modela a relação entre tempo de leitura e conversão no blog.")

    if over_f.empty:
        st.warning("⚠️ Sem dados no período selecionado para os modelos de IA.")
        st.stop()

    st.markdown("---")
    st.header("1. Radar de Anomalias (Z-Score Temporal)")
    explain("O que é Z-Score?",
            "O Z-Score mede quantos desvios padrão um dia está acima ou abaixo da média. "
            "Z > 1.0 = dia viral (estatisticamente acima do normal). Z < -1.0 = dia frio. "
            "Isso permite identificar conteúdos que realmente saíram do padrão, "
            "filtrando ruído do dia a dia.")

    evo = over_f.groupby('Data')['Tração'].sum().reset_index()
    if len(evo) > 2:
        evo['Z-Score'] = stats.zscore(evo['Tração'])
        evo['Status'] = evo['Z-Score'].apply(
            lambda z: '🚀 Viral' if z > 1.0 else ('🧊 Frio' if z < -1.0 else '📊 Normal')
        )
        CORES_Z = {'🚀 Viral': '#10B981', '📊 Normal': '#0094A4', '🧊 Frio': '#DC2626'}
        fig_z = px.bar(evo, x='Data', y='Tração', color='Status',
                       color_discrete_map=CORES_Z,
                       hover_data=['Z-Score'])
        fig_z.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_z, width="stretch")

        dias_virais = evo[evo['Z-Score'] > 1.0]
        dias_frios = evo[evo['Z-Score'] < -1.0]
        col_v, col_f = st.columns(2)
        with col_v:
            st.success(f"🚀 {len(dias_virais)} Dia(s) Viral(is) detectado(s)")
            if not dias_virais.empty:
                virais_content = lista_f[lista_f['Data'].isin(dias_virais['Data'])].sort_values('Cliques/Tração', ascending=False)
                st.dataframe(virais_content[['Data', 'Plataforma', 'Título', 'Cliques/Tração']], hide_index=True, width="stretch")
        with col_f:
            st.error(f"🧊 {len(dias_frios)} Dia(s) Frio(s) detectado(s)")
            if not dias_frios.empty:
                frios_content = lista_f[lista_f['Data'].isin(dias_frios['Data'])].sort_values('Cliques/Tração')
                st.dataframe(frios_content[['Data', 'Plataforma', 'Título', 'Cliques/Tração']], hide_index=True, width="stretch")
    else:
        st.info("Expanda o filtro de datas para ter mais pontos e ativar a análise de anomalias (mínimo 3 dias).")

    st.markdown("---")
    st.header("2. Clusterização de Campanhas (K-Means)")
    explain("Como o K-Means funciona?",
            "O algoritmo agrupa automaticamente as campanhas em 3 clusters baseado em Abertura + CTOR. "
            "Sem supervisão humana, ele encontra padrões escondidos nos dados. "
            "Os rótulos (Estrela, Média, Baixa Atenção) são atribuídos depois, pelo desempenho relativo de cada grupo.")

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
        fig_km = px.scatter(
            mai_km, x='Taxa de Abertura', y='CTOR',
            color='Classificação IA', size='Qtd Enviados',
            hover_name='Título', color_discrete_map=CORES_KM
        )
        fig_km.update_layout(**PLOTLY_LAYOUT, xaxis_tickformat='.1%', yaxis_tickformat='.1%')
        st.plotly_chart(fig_km, width="stretch")

        col_a, col_b, col_c = st.columns(3)
        for col, label in zip([col_a, col_b, col_c], ['🚀 Estrela', '📉 Média', '⚠️ Baixa Atenção']):
            grupo = mai_km[mai_km['Classificação IA'] == label]
            col.metric(label, f"{len(grupo)} campanhas",
                    f"Abertura média: {grupo['Taxa de Abertura'].mean()*100:.1f}%")
    else:
        st.info("Mínimo de 4 campanhas de e-mail necessárias para treinar o K-Means.")

    st.markdown("---")
    st.header("3. Previsibilidade de Conversão (Regressão OLS)")
    explain("O que a regressão diz?",
            "A linha de tendência mostra se existe uma relação entre tempo de leitura e taxa de conversão. "
            "Se a linha sobe da esquerda para direita, artigos mais longos convertem mais no seu caso — "
            "priorize conteúdo denso. Se não há correlação, outros fatores dominam (ex: posição do CTA).")

    if len(blo_f) > 2:
        import plotly.express as px
        fig_ols = px.scatter(
            blo_f, x='Tempo da Página', y='Taxa Conversão',
            size='Views', color='Tag Produto',
            hover_name='Título', color_discrete_map=CORES_PRODUTOS,
            trendline='ols', trendline_scope='overall',
            trendline_color_override='#F97316'
        )
        fig_ols.update_layout(**PLOTLY_LAYOUT, yaxis_tickformat='.1%')
        st.plotly_chart(fig_ols, width="stretch")

        try:
            import statsmodels.api as sm
            X_ols = sm.add_constant(blo_f['Tempo da Página'].fillna(0))
            y_ols = blo_f['Taxa Conversão'].fillna(0)
            modelo = sm.OLS(y_ols, X_ols).fit()
            r2 = modelo.rsquared
            coef = modelo.params.iloc[1]
            st.markdown(f"""
            <div class="info-card">
            <p>📐 <strong>R² = {r2:.3f}</strong> — o modelo explica {r2*100:.1f}% da variação de conversão pelo tempo na página.<br>
            📈 <strong>Coeficiente = {coef:.6f}</strong> — cada segundo a mais de leitura {'aumenta' if coef > 0 else 'diminui'} a taxa de conversão em {abs(coef)*100:.4f}%.</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass
    else:
        st.info("Mínimo de 3 artigos no Blog para traçar a Regressão Linear.")

    st.markdown("---")
    st.header("4. Matriz de Densidade — LinkedIn")
    explain("Combinação ideal",
            "Cada célula mostra o engajamento médio da combinação Tipo × Tamanho. "
            "Cores mais escuras = melhor performance. Use para definir o formato padrão dos próximos posts.")

    ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
    if not ab.empty:
        fig_hm = px.density_heatmap(ab, x='Tipo', y='Tamanho', z='Engajamento',
                                    text_auto='.1f', color_continuous_scale='Teal')
        fig_hm.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hm, width="stretch")
    else:
        st.info("Sem dados do LinkedIn disponíveis para renderizar a matriz.")
