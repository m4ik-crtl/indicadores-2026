import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
import datetime

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

def f_br(num, is_pct=False):
    if pd.isna(num): return "0"
    if is_pct: 
        return f"{num * 100:.2f}%".replace('.', ',')
    else:
        return f"{num:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# ==========================================
# 🧠 MOTOR DE DADOS & CACHE INTELIGENTE
# ==========================================
# 🛡️ ttl=60 obriga o Streamlit a revalidar os dados a cada 1 minuto!
@st.cache_data(ttl=60) 
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
# SIDEBAR / FILTROS BLINDADOS
# ==========================================
st.sidebar.image("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyBpZD0iTGF5ZXJfMiIgZGF0YS1uYW1lPSJMYXllciAyIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzODcuNDQgMTY2LjA1Ij4KICA8ZGVmcz4KICAgIDxzdHlsZT4KICAgICAgLmNscy0xIHsKICAgICAgICBmaWxsOiAjMDA5NGE0OwogICAgICB9CgogICAgICAuY2xzLTIgewogICAgICAgIGZpbGw6IGdyYXk7CiAgICAgIH0KICAgIDwvc3R5bGU+CiAgPC9kZWZzPgogIDxnIGlkPSJMYXllcl8yLTIiIGRhdGEtbmFtZT0iTGF5ZXIgMiI+CiAgICA8Zz4KICAgICAgPGc+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMzc5Ljg4LDM2LjY1Yy01LjMyLTUuMzItMTEuOTMtNy44MS0xOS40OC03Ljgxcy0xNC4wOCwyLjQ5LTE5LjQsNy45Yy01LjQ5LDUuNDktNy44MSwxMi4zNi03LjgxLDIwdjQxLjExaDEwLjgydi00MC41MWMwLTQuODksMS4zNy05LjM2LDQuNzItMTIuOTYsMy4xOC0zLjM1LDYuOTUtNC45OCwxMS41OS00Ljk4czguNDEsMS41NSwxMS41OSw0Ljk4YzMuNDMsMy42MSw0LjcyLDguMDcsNC43MiwxMi45NnY0MC41MWgxMC44MnYtNDEuMTFjLjE3LTcuNjQtMi4xNS0xNC41OS03LjY0LTIwLjA5LDAsMCwuMDksMCwuMDksMFoiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yOTEuMywyNy4xMmMtMTkuNTcsMC0zNS42MiwxNS45Ny0zNS42MiwzNS42MnMxNS45NywzNS42MiwzNS42MiwzNS42MiwzNS42Mi0xNS45NywzNS42Mi0zNS42Mi0xNS45Ny0zNS42Mi0zNS42Mi0zNS42MlpNMjkxLjMsODYuODZjLTEzLjMsMC0yNC4wMy0xMC44Mi0yNC4wMy0yNC4wM3MxMC44Mi0yNC4wMywyNC4wMy0yNC4wMywyNC4wMywxMC44MiwyNC4wMywyNC4wMy0xMC44MiwyNC4wMy0yNC4wMywyNC4wM1oiLz4KICAgICAgICA8cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yMTYuOCwyOC44NGMtOS4xLDAtMTYuOTEsMy4zNS0yMy4wOSw5Ljk2LTYuNTIsNi44Ny05LjI3LDE1LjE5LTkuMjcsMjQuNTVzMi44MywxNy43Nyw5LjM2LDI0LjYzYzYuMjcsNi41MiwxNC4wOCw5Ljg3LDIzLjA5LDkuODdzNi42MS0uNDMsOS43LTEuNTV2LTExLjQyYy0uMjYuMTctLjYuMjYtLjg2LjQzLTIuODMsMS4zNy01Ljg0LDEuODktOC45MywxLjg5LTYuMDEsMC0xMC43My0yLjU4LTE0Ljc2LTYuODctNC41NS00LjcyLTYuNTItMTAuNTYtNi41Mi0xNy4wOHMyLjA2LTEyLjM2LDYuNTItMTcuMDhjNC4wMy00LjI5LDguODQtNi44NywxNC43Ni02Ljg3czExLjMzLDIuMjMsMTUuNDUsNi45NWM0LjI5LDQuODksNi4xOCwxMC41Niw2LjE4LDE3djYyLjE0aDEwLjgydi02Mi4yM2MwLTkuMzYtMi44My0xNy43Ny05LjM2LTI0LjYzLTYuMDktNi41Mi0xMy45MS05Ljg3LTIzLTkuODdoMGwtLjA5LjE3aDBaIi8+CiAgICAgICAgPHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMTY1LjMsMjguODRoMTAuODJ2NjkuMDFoLTEwLjgyVjI4Ljg0WiIvPgogICAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTEyNC43LDI4Ljg0Yy05LjEsMC0xNi45MSwzLjM1LTIzLjA5LDkuOTYtNi40NCw2Ljg3LTkuMjcsMTUuMTktOS4yNywyNC41NXMyLjgzLDE3Ljc3LDkuMzYsMjQuNjNjNi4yNyw2LjUyLDE0LjA4LDkuODcsMjMuMDksOS44N3M2LjUyLS40Myw5LjYxLTEuNTV2LTExLjMzYy0uMjYuMTctLjUyLjI2LS43Ny40My0yLjgzLDEuMzctNS44NCwxLjg5LTguOTMsMS44OS02LjAxLDAtMTAuNzMtMi41OC0xNC43Ni02Ljg3LTQuNTUtNC43Mi02LjUyLTEwLjU2LTYuNTItMTcuMDhzMS45Ny0xMi4yNyw2LjUyLTE3LjA4YzQuMDMtNC4yOSw4Ljg0LTYuODcsMTQuNzYtNi44N3MxMS4zMywyLjIzLDE1LjQ1LDYuOTVjNC4yMSw0Ljg5LDYuMTgsMTAuNTYsNi4xOCwxN3YxMC44MmgwdjE0Ljg1aDB2OC43NmgxMC44MnYtMzQuNTFjMC05LjM2LTIuODMtMTcuNzctOS4zNi0yNC42My02LjE4LTYuNTItMTMuOTktOS44Ny0yMy05Ljg3aDBsLS4wOS4wOWgwWiIvPgogICAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTkuNTEsMzUuMDJjLTUuMDYsMTcuNDIsMjYuNDQsMzAuMzksMzguOCw1MS4wNywxMi4xLDIwLjM0LTEuNzIsMzEuMDctMTEuODUsMzkuNDgsNC4wMy0xMy4zLTEyLjI3LTIxLjgtMjYuMzUtMzcuNTEtMTMuOTktMTUuMjgtMTIuNzktMzYuMjItLjUyLTUzLjA1LDAsMC0uMDksMC0uMDksMFoiLz4KICAgICAgPC9nPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik00NC44NywwYy02LjE4LDIxLjM3LDM2LjkxLDMwLjIxLDM2LjY1LDYxLjgsMCw5Ljg3LTQuODEsMTguNzEtMTEuNjcsMjkuMSw1Ljg0LTExLjg1LTE0LjE2LTI2LjUyLTI4LjQxLTQyLjA2LTEzLjkxLTE1LjI4LTIxLjgtMzIuNjIsMy40My00OC44NFoiLz4KICAgIDwvZz4KICAgIDxnPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMTEuNTIsMTYwLjE0aC0yLjg3di0xNS4wOWgtNS4xNnYtMi40NmgxMy4xOXYyLjQ2aC01LjE2djE1LjA5WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMzAuNzUsMTYwLjE0aC0yLjgzdi04LjE2YzAtMS4wMi0uMjEtMS43OS0uNjItMi4yOXMtMS4wNy0uNzYtMS45Ni0uNzZjLTEuMTgsMC0yLjA1LjM1LTIuNjEsMS4wNi0uNTYuNzEtLjgzLDEuODktLjgzLDMuNTZ2Ni41OWgtMi44MnYtMTguNjhoMi44MnY0Ljc0YzAsLjc2LS4wNSwxLjU3LS4xNCwyLjQ0aC4xOGMuMzgtLjY0LjkyLTEuMTQsMS42LTEuNDkuNjgtLjM1LDEuNDgtLjUzLDIuMzktLjUzLDMuMjIsMCw0LjgyLDEuNjIsNC44Miw0Ljg2djguNjVaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTE0MC40NywxNjAuMzhjLTIuMDYsMC0zLjY4LS42LTQuODQtMS44MXMtMS43NS0yLjg2LTEuNzUtNC45Ny41NC0zLjg3LDEuNjItNS4xMSwyLjU2LTEuODYsNC40NS0xLjg2YzEuNzUsMCwzLjE0LjUzLDQuMTUsMS42LDEuMDIsMS4wNiwxLjUyLDIuNTMsMS41Miw0LjM5djEuNTJoLTguODVjLjA0LDEuMjkuMzksMi4yOCwxLjA0LDIuOTcuNjYuNjksMS41OCwxLjA0LDIuNzcsMS4wNC43OCwwLDEuNTEtLjA3LDIuMTktLjIyLjY4LS4xNSwxLjQtLjM5LDIuMTgtLjc0djIuMjljLS42OS4zMy0xLjM4LjU2LTIuMDkuN3MtMS41MS4yLTIuNDEuMlpNMTM5Ljk1LDE0OC43N2MtLjksMC0xLjYxLjI4LTIuMTYuODUtLjU0LjU3LS44NiwxLjQtLjk3LDIuNDhoNi4wM2MtLjAyLTEuMS0uMjgtMS45My0uNzktMi40OS0uNTEtLjU2LTEuMjItLjg1LTIuMTEtLjg1WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xNTUuNTIsMTQyLjZoNS4yMWMyLjQyLDAsNC4xNi4zNSw1LjI0LDEuMDZzMS42MSwxLjgyLDEuNjEsMy4zNGMwLDEuMDItLjI2LDEuODgtLjc5LDIuNTYtLjUzLjY4LTEuMjksMS4xMS0yLjI4LDEuMjh2LjEyYzEuMjMuMjMsMi4xNC42OSwyLjcyLDEuMzcuNTguNjguODcsMS42MS44NywyLjc4LDAsMS41OC0uNTUsMi44MS0xLjY1LDMuNy0xLjEuODktMi42MywxLjM0LTQuNTksMS4zNGgtNi4zNHYtMTcuNTVaTTE1OC4zOSwxNDkuODVoMi43NmMxLjIsMCwyLjA4LS4xOSwyLjYzLS41N3MuODMtMS4wMy44My0xLjk0YzAtLjgyLS4zLTEuNDItLjg5LTEuNzktLjYtLjM3LTEuNTQtLjU1LTIuODQtLjU1aC0yLjQ5djQuODVaTTE1OC4zOSwxNTIuMTd2NS41NmgzLjA1YzEuMiwwLDIuMTEtLjIzLDIuNzItLjY5cy45Mi0xLjE5LjkyLTIuMThjMC0uOTEtLjMxLTEuNTktLjk0LTIuMDMtLjYyLS40NC0xLjU3LS42Ni0yLjg0LS42NmgtMi45MVoiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMTc4LjE4LDE0Ni42M2MuNTcsMCwxLjA0LjA0LDEuNC4xMmwtLjI4LDIuNjNjLS40LS4xLS44Mi0uMTQtMS4yNS0uMTQtMS4xMywwLTIuMDQuMzctMi43NCwxLjFzLTEuMDUsMS42OS0xLjA1LDIuODd2Ni45NGgtMi44MnYtMTMuMjdoMi4yMWwuMzcsMi4zNGguMTRjLjQ0LS43OSwxLjAxLTEuNDIsMS43Mi0xLjg4LjcxLS40NiwxLjQ3LS43LDIuMjktLjdaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTE5MC4zNSwxNjAuMTRsLS41Ni0xLjg1aC0uMWMtLjY0LjgxLTEuMjgsMS4zNi0xLjkzLDEuNjVzLTEuNDguNDQtMi41LjQ0Yy0xLjMsMC0yLjMyLS4zNS0zLjA1LTEuMDYtLjczLS43LTEuMS0xLjctMS4xLTIuOTksMC0xLjM3LjUxLTIuNCwxLjUyLTMuMSwxLjAyLS43LDIuNTYtMS4wOCw0LjY0LTEuMTRsMi4yOS0uMDd2LS43MWMwLS44NS0uMi0xLjQ4LS41OS0xLjktLjQtLjQyLTEuMDEtLjYzLTEuODQtLjYzLS42OCwwLTEuMzMuMS0xLjk2LjMtLjYyLjItMS4yMi40NC0xLjguNzFsLS45MS0yLjAyYy43Mi0uMzgsMS41MS0uNjYsMi4zNi0uODZzMS42Ni0uMjksMi40Mi0uMjljMS42OSwwLDIuOTYuMzcsMy44MiwxLjFzMS4yOSwxLjg5LDEuMjksMy40N3Y4Ljk0aC0yLjAyWk0xODYuMTUsMTU4LjIyYzEuMDIsMCwxLjg1LS4yOSwyLjQ3LS44NnMuOTMtMS4zOC45My0yLjQxdi0xLjE1bC0xLjcuMDdjLTEuMzMuMDUtMi4yOS4yNy0yLjkuNjctLjYuNC0uOTEsMS0uOTEsMS44MiwwLC41OS4xOCwxLjA1LjUzLDEuMzcuMzUuMzIuODguNDksMS41OC40OVoiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjA1LjE3LDE2MC4xNGgtMTAuMDd2LTEuNzRsNi43MS05LjM3aC02LjN2LTIuMTZoOS40N3YxLjk3bC02LjU3LDkuMTVoNi43NnYyLjE2WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0yMDcuODksMTQzLjM1YzAtLjUuMTQtLjg5LjQxLTEuMTcuMjgtLjI3LjY3LS40MSwxLjE4LS40MXMuODguMTQsMS4xNi40MWMuMjguMjcuNDEuNjYuNDEsMS4xN3MtLjE0Ljg2LS40MSwxLjEzYy0uMjguMjgtLjY2LjQxLTEuMTYuNDFzLS45MS0uMTQtMS4xOC0uNDFjLS4yOC0uMjgtLjQxLS42NS0uNDEtMS4xM1pNMjEwLjg3LDE2MC4xNGgtMi44MnYtMTMuMjdoMi44MnYxMy4yN1oiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjE3LjczLDE2MC4xNGgtMi44MnYtMTguNjhoMi44MnYxOC42OFoiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjIxLjU5LDE0My4zNWMwLS41LjE0LS44OS40MS0xLjE3LjI4LS4yNy42Ny0uNDEsMS4xOC0uNDFzLjg4LjE0LDEuMTYuNDFjLjI4LjI3LjQxLjY2LjQxLDEuMTdzLS4xNC44Ni0uNDEsMS4xM2MtLjI4LjI4LS42Ni40MS0xLjE2LjQxcy0uOTEtLjE0LTEuMTgtLjQxYy0uMjgtLjI4LS40MS0uNjUtLjQxLTEuMTNaTTIyNC41OCwxNjAuMTRoLTIuODJ2LTEzLjI3aDIuODJ2MTMuMjdaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTIzNi45MiwxNjAuMTRsLS41Ni0xLjg1aC0uMWMtLjY0LjgxLTEuMjgsMS4zNi0xLjkzLDEuNjVzLTEuNDguNDQtMi41LjQ0Yy0xLjMsMC0yLjMyLS4zNS0zLjA1LTEuMDYtLjczLS43LTEuMS0xLjctMS4xLTIuOTksMC0xLjM3LjUxLTIuNCwxLjUyLTMuMSwxLjAyLS43LDIuNTYtMS4wOCw0LjY0LTEuMTRsMi4yOS0uMDd2LS43MWMwLS44NS0uMi0xLjQ4LS41OS0xLjktLjQtLjQyLTEuMDEtLjYzLTEuODQtLjYzLS42OCwwLTEuMzMuMS0xLjk2LjMtLjYyLjItMS4yMi40NC0xLjguNzFsLS45MS0yLjAyYy43Mi0uMzgsMS41MS0uNjYsMi4zNi0uODZzMS42Ni0uMjksMi40Mi0uMjljMS42OSwwLDIuOTYuMzcsMy44MiwxLjFzMS4yOSwxLjg5LDEuMjksMy40N3Y4Ljk0aC0yLjAyWk0yMzIuNzIsMTU4LjIyYzEuMDIsMCwxLjg1LS4yOSwyLjQ3LS44NnMuOTMtMS4zOC45My0yLjQxdi0xLjE1bC0xLjcuMDdjLTEuMzMuMDUtMi4yOS4yNy0yLjkuNjctLjYuNC0uOTEsMS0uOTEsMS44MiwwLC41OS4xOCwxLjA1LjUzLDEuMzcuMzUuMzIuODguNDksMS41OC40OVoiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjU0LjU1LDE2MC4xNGgtMi44M3YtOC4xNmMwLTEuMDItLjIxLTEuNzktLjYyLTIuMjlzLTEuMDctLjc2LTEuOTYtLjc2Yy0xLjE5LDAtMi4wNi4zNS0yLjYyLDEuMDZzLS44MywxLjg4LS44MywzLjU0djYuNjFoLTIuODJ2LTEzLjI3aDIuMjFsLjQsMS43NGguMTRjLjQtLjYzLjk3LTEuMTIsMS43MS0xLjQ2Ljc0LS4zNCwxLjU1LS41MiwyLjQ1LS41MiwzLjE4LDAsNC43OCwxLjYyLDQuNzgsNC44NnY4LjY1WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0yNzIuNjMsMTQ0LjgxYy0xLjY1LDAtMi45NS41OC0zLjg5LDEuNzUtLjk0LDEuMTctMS40MiwyLjc4LTEuNDIsNC44NHMuNDUsMy43OCwxLjM2LDQuODhjLjkxLDEuMSwyLjIyLDEuNjYsMy45NCwxLjY2Ljc0LDAsMS40Ni0uMDcsMi4xNi0uMjIuNy0uMTUsMS40Mi0uMzQsMi4xNy0uNTd2Mi40NmMtMS4zOC41Mi0yLjk0Ljc4LTQuNjguNzgtMi41NywwLTQuNTQtLjc4LTUuOTItMi4zMy0xLjM4LTEuNTYtMi4wNi0zLjc4LTIuMDYtNi42OCwwLTEuODIuMzMtMy40MiwxLTQuNzlzMS42My0yLjQyLDIuOS0zLjE0YzEuMjYtLjczLDIuNzUtMS4wOSw0LjQ1LTEuMDksMS43OSwwLDMuNDUuMzgsNC45NywxLjEzbC0xLjAzLDIuMzljLS41OS0uMjgtMS4yMi0uNTMtMS44OC0uNzQtLjY2LS4yMS0xLjM1LS4zMi0yLjA4LS4zMloiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjc4LjQyLDE0Ni44N2gzLjA3bDIuNyw3LjUzYy40MSwxLjA3LjY4LDIuMDguODIsMy4wMmguMWMuMDctLjQ0LjItLjk3LjQtMS42LjE5LS42MywxLjIxLTMuNjEsMy4wNS04Ljk1aDMuMDVsLTUuNjgsMTUuMDRjLTEuMDMsMi43Ni0yLjc1LDQuMTQtNS4xNiw0LjE0LS42MiwwLTEuMjMtLjA3LTEuODMtLjJ2LTIuMjNjLjQyLjEuOTEuMTQsMS40NS4xNCwxLjM2LDAsMi4zMi0uNzksMi44Ny0yLjM2bC40OS0xLjI1LTUuMzMtMTMuMjdaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTMwMC40LDE0Ni42M2MxLjY2LDAsMi45NS42LDMuODcsMS44LjkyLDEuMiwxLjM5LDIuODgsMS4zOSw1LjA1cy0uNDcsMy44Ny0xLjQsNS4wOGMtLjk0LDEuMjEtMi4yNCwxLjgyLTMuOSwxLjgycy0yLjk5LS42LTMuOTEtMS44MWgtLjE5bC0uNTIsMS41N2gtMi4xMXYtMTguNjhoMi44MnY0LjQ0YzAsLjMzLS4wMi44Mi0uMDUsMS40NnMtLjA2LDEuMDYtLjA3LDEuMjRoLjEyYy45LTEuMzIsMi4yMi0xLjk4LDMuOTYtMS45OFpNMjk5LjY3LDE0OC45M2MtMS4xNCwwLTEuOTUuMzMtMi40NSwxcy0uNzYsMS43OS0uNzcsMy4zNXYuMTljMCwxLjYyLjI2LDIuNzkuNzcsMy41MS41MS43MiwxLjM1LDEuMDksMi41MSwxLjA5LDEsMCwxLjc2LS40LDIuMjctMS4xOS41Mi0uNzkuNzctMS45NC43Ny0zLjQzLDAtMy4wMi0xLjAzLTQuNTItMy4xLTQuNTJaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTMxNC43MywxNjAuMzhjLTIuMDYsMC0zLjY4LS42LTQuODQtMS44MXMtMS43NS0yLjg2LTEuNzUtNC45Ny41NC0zLjg3LDEuNjItNS4xMSwyLjU2LTEuODYsNC40NS0xLjg2YzEuNzUsMCwzLjE0LjUzLDQuMTUsMS42LDEuMDIsMS4wNiwxLjUyLDIuNTMsMS41Miw0LjM5djEuNTJoLTguODVjLjA0LDEuMjkuMzksMi4yOCwxLjA0LDIuOTcuNjYuNjksMS41OCwxLjA0LDIuNzcsMS4wNC43OCwwLDEuNTEtLjA3LDIuMTktLjIyLjY4LS4xNSwxLjQtLjM5LDIuMTgtLjc0djIuMjljLS42OS4zMy0xLjM4LjU2LTIuMDkuN3MtMS41MS4yLTIuNDEuMlpNMzE0LjIyLDE0OC43N2MtLjksMC0xLjYxLjI4LTIuMTYuODUtLjU0LjU3LS44NiwxLjQtLjk3LDIuNDhoNi4wM2MtLjAyLTEuMS0uMjgtMS45My0uNzktMi40OS0uNTEtLjU2LTEuMjItLjg1LTIuMTEtLjg1WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0zMjkuODMsMTQ2LjYzYy41NywwLDEuMDQuMDQsMS40LjEybC0uMjgsMi42M2MtLjQtLjEtLjgyLS4xNC0xLjI1LS4xNC0xLjEzLDAtMi4wNC4zNy0yLjc0LDEuMXMtMS4wNSwxLjY5LTEuMDUsMi44N3Y2Ljk0aC0yLjgydi0xMy4yN2gyLjIxbC4zNywyLjM0aC4xNGMuNDQtLjc5LDEuMDEtMS40MiwxLjcyLTEuODguNzEtLjQ2LDEuNDctLjcsMi4yOS0uN1oiLz4KICAgICAgPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMzU0LjIyLDE2MC4xNGgtMi44OHYtNy45MWgtOC4wOXY3LjkxaC0yLjg3di0xNy41NWgyLjg3djcuMThoOC4wOXYtNy4xOGgyLjg4djE3LjU1WiIvPgogICAgICA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0zNjcuOSwxNjAuMTRsLS40LTEuNzRoLS4xNGMtLjM5LjYyLS45NSwxLjEtMS42NywxLjQ1LS43Mi4zNS0xLjU1LjUzLTIuNDguNTMtMS42MSwwLTIuODEtLjQtMy42LTEuMi0uNzktLjgtMS4xOS0yLjAxLTEuMTktMy42NHYtOC42OGgyLjg0djguMTljMCwxLjAyLjIxLDEuNzguNjIsMi4yOS40Mi41MSwxLjA3Ljc2LDEuOTYuNzYsMS4xOCwwLDIuMDUtLjM1LDIuNjEtMS4wNi41Ni0uNzEuODMtMS44OS44My0zLjU2di02LjYxaDIuODN2MTMuMjdoLTIuMjJaIi8+CiAgICAgIDxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTM4MC45MiwxNDYuNjNjMS42NiwwLDIuOTUuNiwzLjg3LDEuOC45MiwxLjIsMS4zOSwyLjg4LDEuMzksNS4wNXMtLjQ3LDMuODctMS40LDUuMDhjLS45NCwxLjIxLTIuMjQsMS44Mi0zLjksMS44MnMtMi45OS0uNi0zLjkxLTEuODFoLS4xOWwtLjUyLDEuNTdoLTIuMTF2LTE4LjY4aDIuODJ2NC40NGMwLC4zMy0uMDIuODItLjA1LDEuNDZzLS4wNiwxLjA2LS4wNywxLjI0aC4xMmMuOS0xLjMyLDIuMjItMS45OCwzLjk2LTEuOThaTTM4MC4xOSwxNDguOTNjLTEuMTQsMC0xLjk1LjMzLTIuNDUsMXMtLjc2LDEuNzktLjc3LDMuMzV2LjE5YzAsMS42Mi4yNiwyLjc5Ljc3LDMuNTEuNTEuNzIsMS4zNSwxLjA5LDIuNTEsMS4wOSwxLDAsMS43Ni0uNCwyLjI3LTEuMTkuNTItLjc5Ljc3LTEuOTQuNzctMy40MywwLTMuMDItMS4wMy00LjUyLTMuMS00LjUyWiIvPgogICAgPC9nPgogIDwvZz4KPC9zdmc+", width=100)
st.sidebar.title("AIQON Inteligência")
menu = st.sidebar.radio("Navegação:", ["🌐 Full Blast (Overview)", "💼 LinkedIn (ML Lab)", "📧 Mailchimp (Funil)", "📝 Blog (SEO & OLS)", "🤖 IA & Machine Learning"])
st.sidebar.markdown("---")

# 🛡️ BUSCA DE DATAS SEGURA (Ignora Nulos/Vazios para evitar falhas)
todas_datas = pd.Series(
    df_over['Data'].dropna().tolist() + 
    df_lin['Data'].dropna().tolist() + 
    df_blo['Data'].dropna().tolist() + 
    df_mai['Data de Envio'].dropna().tolist()
)

if not todas_datas.empty:
    min_d = todas_datas.min()
    max_d = todas_datas.max()
else:
    min_d = datetime.date.today()
    max_d = datetime.date.today()

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

if menu == "🌐 Full Blast (Overview)":
    st.title("Impacto da Marca (Full Blast)")
    
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

elif menu == "💼 LinkedIn (ML Lab)":
    st.title("LinkedIn: Desempenho e IA")
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

elif menu == "📧 Mailchimp (Funil)":
    st.title("Mailchimp: Conversão")
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
        st.subheader("Abertura vs CTOR")
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

elif menu == "📝 Blog (SEO & OLS)":
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
        st.markdown("A IA analisa a série histórica para separar os conteúdos que **viralizaram** daqueles que foram para a **geladeira**.")
        
        evo = over_f.groupby('Data')['Tração'].sum().reset_index()
        if len(evo) > 2:
            z_scores = stats.zscore(evo['Tração'])
            evo['Z-Score'] = z_scores
            
            dias_virais = evo[evo['Z-Score'] > 1.0]
            dias_frios = evo[evo['Z-Score'] < -1.0]
            
            col_v, col_f = st.columns(2)
            with col_v:
                st.success(f"🚀 **{len(dias_virais)} Pico(s) Viral(is)** detectado(s).")
                if not dias_virais.empty:
                    lista_virais = lista_f[lista_f['Data'].isin(dias_virais['Data'])].sort_values('Cliques/Tração', ascending=False)
                    st.dataframe(lista_virais[['Data', 'Plataforma', 'Título']], hide_index=True, use_container_width=True)
            
            with col_f:
                st.error(f"🧊 **{len(dias_frios)} Dia(s) Frio(s)** detectado(s).")
                if not dias_frios.empty:
                    lista_frios = lista_f[lista_f['Data'].isin(dias_frios['Data'])].sort_values('Cliques/Tração', ascending=True)
                    st.dataframe(lista_frios[['Data', 'Plataforma', 'Título']], hide_index=True, use_container_width=True)
        else:
            st.info("Volume de dias insuficiente para calcular anomalias estáticas. Expanda o filtro de datas.")
            
        st.markdown("---")

        # 2. KMeans (Clusterização de E-mail)
        st.header("2. Clusterização de Audiência (K-Means)")
        
        if len(mai_f) > 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            X = mai_f[['Taxa de Abertura', 'CTOR']].fillna(0)
            mai_f['Cluster (IA)'] = kmeans.fit_predict(X)
            centroides = kmeans.cluster_centers_
            soma_centroides = centroides.sum(axis=1)
            ranking_clusters = soma_centroides.argsort()
            mapa = {ranking_clusters[0]: 'Baixa Atenção ⚠️', ranking_clusters[1]: 'Média 📉', ranking_clusters[2]: 'Estrela 🚀'}
            mai_f['Classificação IA'] = mai_f['Cluster (IA)'].map(mapa)

            fig_km = px.scatter(mai_f, x='Taxa de Abertura', y='CTOR', color='Classificação IA', size='Qtd Enviados', hover_name='Título',
                                color_discrete_map={'Estrela 🚀': '#10B981', 'Média 📉': '#F59E0B', 'Baixa Atenção ⚠️': '#DC2626'})
            fig_km.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
            st.plotly_chart(fig_km, use_container_width=True)
        else:
            st.info("E-mails insuficientes para treinar o K-Means (Mín. 4 campanhas).")

        st.markdown("---")

        # 3. OLS Regression (Regressão no Blog)
        st.header("3. Previsibilidade de Conversão (OLS Regression)")
        if len(blo_f) > 2:
            fig_ols = px.scatter(blo_f, x='Tempo da Página', y='Taxa Conversão', size='Views', color='Tag Produto', hover_name='Título', 
                                 color_discrete_map=CORES_PRODUTOS, trendline="ols", trendline_scope="overall")
            fig_ols.update_traces(hovertemplate="Tempo: %{x}s<br>Conversão: %{y:.2%}<extra></extra>")
            st.plotly_chart(fig_ols, use_container_width=True)
        else:
            st.info("Dados insuficientes do Blog para traçar a Regressão Linear.")

        st.markdown("---")

        # 4. Feature Matrix (LinkedIn)
        st.header("4. Matriz de Densidade (LinkedIn)")
        ab = lin_f.groupby(['Tipo', 'Tamanho'])['Engajamento'].mean().reset_index()
        if not ab.empty:
            fig_hm = px.density_heatmap(ab, x="Tipo", y="Tamanho", z="Engajamento", text_auto=".1f", color_continuous_scale="Blues")
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Sem dados do LinkedIn para renderizar a matriz.")
