import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

st.set_page_config(
    page_title="SmartNOMAC - INSTAT",
    page_icon="🇲🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Changement global de la police */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    }

    .main { background-color: #f0f2f6; }
    
    .title-container { width: 100%; display: flex; justify-content: center; margin-top: 40px; margin-bottom: 20px; }
    
    .title-text { 
        color: #1E3A8A; 
        font-weight: 800; 
        font-size: 3.5rem; 
        text-align: center;
        letter-spacing: -1.5px;
    }
    
    /* Cartes Dashboard */
    .metric-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 6px solid #1E3A8A;
    }
    .metric-card h3 { 
        color: #1E3A8A !important; 
        font-size: 2.8rem !important; 
        margin: 0 !important;
        font-weight: 800 !important;
    }
    .metric-card p { 
        color: #334155 !important; 
        font-size: 1.1rem !important; 
        font-weight: 600 !important;
        margin: 5px 0 0 0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Styles des textes de confiance */
    .high-confidence { color: #15803d; font-weight: 700; }
    .med-confidence { color: #b45309; font-weight: 700; }
    .low-confidence { color: #b91c1c; font-weight: 700; }
    
    /* Boutons personnalisés */
    div.stButton > button:first-child {
        background-color: #1E3A8A;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        height: 3.2em;
        border: none;
        font-size: 1rem;
    }

    /* Style spécifique pour la Sidebar */
    [data-testid="stSidebarNav"] span {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_secteur_name(code):
    first_digit = str(code)[0]
    mapping = {
        '1': 'Agriculture & Pêche',
        '2': 'Mines & Carrières',
        '3': 'Industries Manufacturières',
        '4': 'Énergie & Eau',
        '5': 'Construction & BTP',
        '6': 'Commerce & Transports',
        '7': 'Hébergement & Restauration',
        '8': 'Services Financiers / Immo',
        '9': 'Administration & Social'
    }
    return mapping.get(first_digit, 'Autres Services')

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown('<div class="title-container"><h1 class="title-text">SmartNOMAC</h1></div>', unsafe_allow_html=True)
        _, center_col, _ = st.columns([1, 0.8, 1])
        with center_col:
            with st.container(border=True):
                st.markdown("<h3 style='text-align:center;'>🔐 Authentification</h3>", unsafe_allow_html=True)
                pwd = st.text_input("Mot de passe", type="password")
                if st.button("Se connecter"):
                    if pwd == "admin123":
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else:
                        st.error("❌ Accès refusé")
        return False
    return True

@st.cache_resource
def load_all():
    try:
        vec = joblib.load('vectorizer.pkl')
        matrix = joblib.load('tfidf_matrix.pkl')
        df = pd.read_pickle('df_nomac.pkl')
        df['Secteur'] = df['Code'].apply(get_secteur_name)
        return vec, matrix, df
    except:
        return None, None, None


if check_password():
    vectorizer, tfidf_matrix, df_final = load_all()
    
    if vectorizer is not None:
        if os.path.exists("logoInstat.jpeg"):
            st.sidebar.markdown("<div style='text-align: center; padding-top: 10px;'>", unsafe_allow_html=True)
            st.sidebar.image("logoInstat.jpeg", width=80)
            st.sidebar.markdown("</div>", unsafe_allow_html=True)
        
        st.sidebar.markdown("<h2 style='text-align: center; color: #1E3A8A; font-weight:800;'>SmartNOMAC</h2>", unsafe_allow_html=True)
        page = st.sidebar.radio("Navigation", ["📊 Tableau de Bord", "🔍 Recherche IA", "📂 Base NOMAC"])
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Déconnexion"):
            st.session_state["password_correct"] = False
            st.rerun()

        
        if page == "📊 Tableau de Bord":
            st.title("📊 Statistiques NOMAC")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><h3>{len(df_final)}</h3><p>Codes Référencés</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><h3>{tfidf_matrix.shape[1]}</h3><p>Mots Indexés</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><h3>9</h3><p>Secteurs</p></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            col_chart, col_info = st.columns([1.6, 1])
            with col_chart:
                sect_counts = df_final['Secteur'].value_counts()
                fig = px.pie(values=sect_counts.values, names=sect_counts.index, 
                             hole=0.5, title="Structure de l'Économie",
                             color_discrete_sequence=px.colors.qualitative.G10)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_info:
                st.markdown("### 📥 Extraction des données")
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Exporter vers Excel (CSV)", data=csv, file_name="base_nomac.csv", mime="text/csv")
                st.info("Cette base est conforme à la nomenclature nationale malgache.")

        
        elif page == "🔍 Recherche IA":
            st.title("🔍 Assistant de Codification")
            query = st.text_input("Saisissez l'activité à classifier :", placeholder="Ex: Commerce de gros de riz")
            
            if query:
                query_p = re.sub(r'[^\w\s]', ' ', query.lower())
                query_vec = vectorizer.transform([query_p])
                scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
                indices = scores.argsort()[-3:][::-1]
                
                for i in indices:
                    score = scores[i]
                    if score > 0.01:
                        row = df_final.iloc[i]
                        pct = round(score*100, 1)
                        conf_class = "high-confidence" if pct > 70 else "med-confidence" if pct > 40 else "low-confidence"
                        with st.expander(f"CODE {row['Code']} | {row['Secteur']}", expanded=True):
                            st.markdown(f"Indice de confiance : <span class='{conf_class}'>{pct}%</span>", unsafe_allow_html=True)
                            st.write(row['Description'])
                            st.button("✅ Valider ce code", key=f"v_{i}")

        
        elif page == "📂 Base NOMAC":
            st.title("📂 Référentiel des Activités")
            secteur_filter = st.selectbox("Filtrer par domaine :", ["Tous les secteurs"] + list(df_final['Secteur'].unique()))
            df_display = df_final if secteur_filter == "Tous les secteurs" else df_final[df_final['Secteur'] == secteur_filter]
            st.dataframe(df_display[['Code', 'Secteur', 'Description']], use_container_width=True, height=500)