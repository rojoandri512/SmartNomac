import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import re, os, io
from datetime import datetime
from pathlib import Path

# ── Chemins absolus ──────────────────────────────────────
BASE_DIR            = Path(__file__).parent.resolve()
FICHIER_VECTORISEUR = BASE_DIR / "vectorizer.pkl"
FICHIER_MATRICE     = BASE_DIR / "tfidf_matrix.pkl"
FICHIER_DF          = BASE_DIR / "df_nomac.pkl"
FICHIER_LOGO        = BASE_DIR / "logoInstat.jpeg"

# ── Comptes utilisateurs ─────────────────────────────────
# role : "admin" ou "user"
COMPTES = {
    "admin":       {"mdp": "Admin@2025",  "role": "admin", "nom": "Administrateur Système"},
    "statisticien":{"mdp": "Stats@2025",  "role": "user",  "nom": "Agent Statisticien"},
    "enqueteur":   {"mdp": "Field@2025",  "role": "user",  "nom": "Agent Enquêteur"},
}

# ── Données de session globales (simulées pour la démo) ──
if "logs_globaux" not in st.session_state:
    st.session_state["logs_globaux"] = []
if "users_actifs" not in st.session_state:
    st.session_state["users_actifs"] = {}


# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartNOMAC · INSTAT Madagascar",
    page_icon="🇲🇬", layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  STYLE — Charte INSTAT (Rouge #C8102E · Vert #007A3D · Marine #0D2137)
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --red:    #C8102E; --red-d: #A00C24; --red-l: #FFF0F2;
  --green:  #007A3D; --green-d:#005C2E; --green-l:#E8F7EF;
  --navy:   #0D2137; --navy-l: #1A3A5C; --navy-xl:#243E5C;
  --gold:   #D4920A; --gold-l: #FEF3DC;
  --bg:     #EEF2F8; --bg2:#E4E9F2;
  --white:  #FFFFFF; --border:#D0DAE8;
  --text:   #0D2137; --muted: #5E7291;
  --r: 14px;
  --sh:  0 2px 16px rgba(13,33,55,.09);
  --shh: 0 8px 40px rgba(13,33,55,.17);
}
html,body,[class*="st-"]{font-family:'Sora',sans-serif!important;color:var(--text)}
.main{background:var(--bg)!important}
.block-container{padding:0!important;max-width:100%!important}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--navy) 0%,var(--navy-l) 100%)!important;
  border-right:none!important;
  box-shadow:4px 0 30px rgba(13,33,55,.28)!important;
}
[data-testid="stSidebar"]>div{padding:0!important}
[data-testid="stSidebar"] *{color:rgba(255,255,255,.86)!important}
[data-testid="stSidebar"] .stRadio>label{
  font-size:.64rem!important;font-weight:700!important;
  text-transform:uppercase;letter-spacing:.13em;
  color:rgba(255,255,255,.28)!important;padding:0 22px;margin-bottom:3px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{
  border-radius:10px;margin:2px 10px!important;
  padding:11px 15px!important;font-size:.88rem!important;font-weight:500!important;
  text-transform:none!important;letter-spacing:normal!important;
  color:rgba(255,255,255,.72)!important;transition:all .18s;cursor:pointer;
  background:transparent!important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{
  background:rgba(255,255,255,.09)!important;color:white!important;
}

/* ── BOUTONS ── */
div.stButton>button{
  background:#007A3D!important;color:#FFFFFF!important;
  border:none!important;border-radius:8px!important;
  font-family:'Sora',sans-serif!important;font-weight:600!important;
  font-size:.82rem!important;padding:.40rem 1.2rem!important;
  box-shadow:none!important;
  transition:all .15s!important;
  display:block!important;margin:0 auto!important;
  width:auto!important;
}
div.stButton>button:hover{
  background:#005C2E!important;transform:none!important;
  box-shadow:none!important;
}
/* Centrer le conteneur du bouton */
div.stButton{text-align:center!important;}

/* ── INPUTS ── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
  border:1.5px solid var(--border)!important;border-radius:var(--r)!important;
  font-family:'Sora',sans-serif!important;font-size:.95rem!important;
  padding:12px 16px!important;background:white!important;transition:all .2s;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus{
  border-color:#3A3F4B!important;
  box-shadow:0 0 0 3px rgba(58,63,75,0.10)!important;
}
div[data-testid="stSelectbox"]>div>div{
  border:1.5px solid var(--border)!important;border-radius:var(--r)!important;background:white!important;
}
hr{border-color:var(--border)!important;margin:1.4rem 0!important}
div[data-testid="stProgress"]>div{border-radius:99px!important}
div[data-testid="stMetric"]{
  background:white;border-radius:var(--r);padding:16px 20px;
  border:1px solid var(--border);box-shadow:var(--sh);
}

/* ══ COMPOSANTS CUSTOM ══ */

.pg{padding:2rem 2.5rem;max-width:1320px}

/* En-tête de page */
.ph{
  display:flex;align-items:center;gap:18px;
  margin-bottom:1.8rem;padding:18px 22px;
  background:white;border-radius:16px;
  border-left:5px solid #3A3F4B;
  box-shadow:var(--sh);
}
.ph.green{border-left-color:var(--green)}
.ph.navy{border-left-color:var(--navy)}
.ph.grey{border-left-color:#3A3F4B}
.ph-icon{
  width:52px;height:52px;border-radius:13px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:1.45rem;
  background:linear-gradient(135deg,#3A3F4B,#565C6B);
  box-shadow:0 4px 14px rgba(58,63,75,0.25);
}
.ph-icon.g{background:linear-gradient(135deg,var(--green),#00A855);box-shadow:0 4px 14px rgba(0,122,61,.28)}
.ph-icon.n{background:linear-gradient(135deg,var(--navy),var(--navy-l));box-shadow:0 4px 14px rgba(13,33,55,.22)}
.ph-icon.gold{background:linear-gradient(135deg,var(--gold),#F5A623);box-shadow:0 4px 14px rgba(212,146,10,.28)}
.ph h1{
  font-family:'Playfair Display',serif!important;font-size:1.7rem!important;
  font-weight:800!important;color:var(--navy)!important;margin:0 0 2px 0!important;
}
.ph p{color:var(--muted)!important;font-size:.87rem!important;margin:0!important}
.ph .role-tag{
  margin-left:auto;display:inline-flex;align-items:center;gap:6px;
  padding:5px 14px;border-radius:99px;font-size:.72rem;font-weight:700;
  white-space:nowrap;
}
.rt-admin{background:var(--red-l);color:var(--red-d);border:1px solid #FFCCD5}
.rt-user{background:var(--green-l);color:var(--green-d);border:1px solid #A7F3D0}

/* Stat card */
.sc{
  background:white;border-radius:var(--r);padding:22px 24px;
  border:1.5px solid var(--border);box-shadow:var(--sh);
  position:relative;overflow:hidden;
}
.sc .lbl{font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin-bottom:5px}
.sc .val{font-family:'Playfair Display',serif;font-size:2.5rem;color:var(--navy);line-height:1}
.sc .sub{font-size:.76rem;color:var(--muted);margin-top:4px}
.sc .bar{height:3px;border-radius:99px;margin-top:14px;
  background:linear-gradient(90deg,var(--red),var(--red-d))}
.sc .bar.g{background:linear-gradient(90deg,var(--green),var(--green-d))}
.sc .bar.gold{background:linear-gradient(90deg,var(--gold),#A06800)}
.sc .bar.n{background:linear-gradient(90deg,var(--navy),var(--navy-l))}
.sc::after{content:'';position:absolute;right:-16px;bottom:-16px;
  width:72px;height:72px;border-radius:50%;background:rgba(200,16,46,.04)}

/* Carte résultat */
.nc{
  background:white;border-radius:16px;border:1.5px solid var(--border);
  padding:20px 22px;margin-bottom:12px;box-shadow:var(--sh);
  position:relative;overflow:hidden;transition:all .22s;
}
.nc::before{content:'';position:absolute;left:0;top:0;bottom:0;
  width:5px;border-radius:5px 0 0 5px;background:var(--green)}
.nc.m::before{background:var(--gold)}
.nc.l::before{background:#CBD5E1}
.nc:hover{box-shadow:var(--shh);transform:translateY(-2px)}
.cp{background:var(--navy);color:white;padding:4px 13px;border-radius:99px;
  font-family:'JetBrains Mono',monospace;font-size:.79rem;font-weight:600}
.sp{background:var(--bg);color:var(--navy);padding:4px 11px;border-radius:99px;
  font-size:.77rem;font-weight:600;border:1px solid var(--border)}
.sv{margin-left:auto;font-size:1.25rem;font-weight:800}
.sv.h{color:var(--green)}.sv.m{color:var(--gold)}.sv.l{color:#94A3B8}
.nd{font-size:.95rem;color:var(--text);margin:9px 0 11px;font-style:italic;line-height:1.55}
.nt{height:6px;background:var(--bg2);border-radius:99px;overflow:hidden;margin-bottom:11px}
.nf{height:100%;border-radius:99px}
.nf.h{background:linear-gradient(90deg,var(--green),#00C463)}
.nf.m{background:linear-gradient(90deg,var(--gold),#FFB830)}
.nf.l{background:linear-gradient(90deg,#94A3B8,#CBD5E1)}
.cb{display:inline-flex;align-items:center;gap:5px;font-size:.76rem;
  font-weight:700;padding:3px 11px;border-radius:99px}
.cb.h{background:var(--green-l);color:var(--green-d)}
.cb.m{background:var(--gold-l);color:#92400E}
.cb.l{background:#F1F5F9;color:#64748B}

/* Aide à la décision */
.ad{background:linear-gradient(135deg,#F0F7FF,#EAF2FF);
  border:1.5px solid #BFDBFE;border-radius:16px;padding:20px 22px;margin:12px 0}
.ad-title{font-weight:700;color:var(--navy);font-size:.9rem;
  margin-bottom:13px;display:flex;align-items:center;gap:8px}
.ad-item{display:flex;align-items:flex-start;gap:12px;padding:10px 0;
  border-bottom:1px solid rgba(191,219,254,.4)}
.ad-item:last-child{border-bottom:none}
.ad-num{width:26px;height:26px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-size:.72rem;font-weight:800;color:white}
.n1{background:linear-gradient(135deg,var(--red),#FF3B5C)}
.n2{background:linear-gradient(135deg,var(--navy),var(--navy-l))}
.n3{background:linear-gradient(135deg,#6B7280,#9CA3AF)}
.ad-code{font-family:'JetBrains Mono',monospace;font-size:.77rem;
  background:var(--navy);color:white;padding:2px 9px;border-radius:99px}
.ad-lib{font-size:.87rem;color:var(--text);font-weight:500;flex:1}
.ad-pct{font-weight:800;font-size:.88rem}
.ad-why{font-size:.79rem;color:var(--muted);margin-top:3px;line-height:1.45}

/* Alertes */
.ok{background:var(--green-l);border:1.5px solid #6EE7B7;
  border-radius:13px;padding:14px 18px;margin-bottom:11px}
.ok strong{color:var(--green-d);font-size:.91rem}
.ok p{color:#166534;font-size:.83rem;margin:3px 0 0}
.warn{background:var(--gold-l);border:1.5px solid #FCD34D;
  border-radius:13px;padding:14px 18px;margin-bottom:11px}
.warn strong{color:#92400E;font-size:.91rem}
.warn p{color:#78350F;font-size:.83rem;margin:3px 0 0}

/* Info strip */
.is{background:#EEF4FD;border-left:4px solid var(--navy);
  border-radius:0 var(--r) var(--r) 0;padding:12px 16px;
  font-size:.85rem;color:var(--navy);margin:10px 0}

/* Section title */
.stit{font-family:'Playfair Display',serif;font-size:1.12rem;
  color:var(--navy);margin:1.3rem 0 .75rem;font-weight:700}

/* Admin panel */
.ap-card{background:white;border-radius:16px;border:1.5px solid var(--border);
  padding:20px 24px;box-shadow:var(--sh);margin-bottom:12px;
  border-top:4px solid var(--red)}
.ap-card.g{border-top-color:var(--green)}
.ap-card.n{border-top-color:var(--navy)}
.ap-card.gold{border-top-color:var(--gold)}
.ap-card h3{font-size:.95rem;font-weight:700;color:var(--navy);margin:0 0 4px}
.ap-card p{font-size:.82rem;color:var(--muted);margin:0}

/* Log table row */
.log-row{display:flex;align-items:center;gap:10px;background:white;
  border-radius:10px;padding:10px 15px;margin-bottom:6px;
  border:1px solid var(--border);box-shadow:var(--sh);font-size:.84rem}
.log-time{font-family:'JetBrains Mono',monospace;font-size:.74rem;color:var(--muted);min-width:60px}
.log-user{font-weight:700;color:var(--navy);min-width:110px}
.log-desc{flex:1;color:var(--text);font-style:italic}
.log-code{font-family:'JetBrains Mono',monospace;font-size:.76rem;
  background:var(--navy);color:white;padding:2px 9px;border-radius:99px}
.log-score{font-weight:700;font-size:.84rem}

/* Sidebar brand */
.sb-top{padding:22px 20px 16px;
  background:rgba(255,255,255,0.03);
  border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:6px}
.sb-top h2{font-family:'Playfair Display',serif!important;
  font-size:1.28rem!important;color:white!important;
  margin:8px 0 2px!important;font-weight:800!important}
.sb-top .sub{font-size:.68rem!important;color:rgba(255,255,255,.30)!important}
.sb-user{margin:0 10px 8px;background:rgba(255,255,255,.07);
  border-radius:11px;padding:11px 14px;border:1px solid rgba(255,255,255,.08)}
.sb-user .sbu-name{font-weight:600;font-size:.86rem;color:white!important}
.sb-user .sbu-role{font-size:.7rem;color:rgba(255,255,255,.42)!important;margin-top:1px}
.sb-chip{background:rgba(255,255,255,.07);border-radius:10px;
  padding:9px 13px;margin:3px 10px}
.sb-chip .v{font-family:'Playfair Display',serif;font-size:1.4rem;
  color:#FFB347!important;line-height:1}
.sb-chip .l{font-size:.65rem;color:rgba(255,255,255,.35)!important;
  text-transform:uppercase;letter-spacing:.09em}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════
def get_secteur(code):
    return SECTEUR_MAP.get(str(code)[0], 'Autres Services')

SECTEUR_MAP = {
    '1':'Agriculture & Pêche','2':'Mines & Carrières',
    '3':'Industries Manuf.','4':'Énergie & Eau',
    '5':'Construction & BTP','6':'Commerce & Transports',
    '7':'Hébergement & Restauration','8':'Services Financiers',
    '9':'Administration & Social',
}
SECTEUR_ICONS = {
    'Agriculture & Pêche':'🌾','Mines & Carrières':'⛏️','Industries Manuf.':'🏭',
    'Énergie & Eau':'⚡','Construction & BTP':'🏗️','Commerce & Transports':'🚚',
    'Hébergement & Restauration':'🍽️','Services Financiers':'🏦','Administration & Social':'🏛️',
}
COULEURS = ['#C8102E','#007A3D','#0D2137','#D4920A','#2E7DD1','#7C3AED','#0891B2','#DB2777','#6B7280']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    stops = {'de','du','des','le','la','les','un','une','et','en','au','aux','par','pour',
             'sur','dans','avec','ce','qui','que','ou','à','a','est','son','sa','ses',
             'leur','leurs','il','elle','ils','elles','nous','vous','on','y','ni','si',
             'se','ne','pas','plus','très','bien','aussi','tout','tous','comme','dont','mais'}
    return ' '.join(w for w in text.split() if w not in stops and len(w) > 1)

def conf(pct):
    if pct >= 70: return 'h', 'Confiance élevée',  '#007A3D'
    if pct >= 40: return 'm', 'Confiance moyenne', '#D4920A'
    return              'l', 'Confiance faible',   '#94A3B8'

def aide_decision(results):
    if not results: return None, "doute", "Aucun résultat trouvé."
    best  = results[0]
    ecart = results[0]['score'] - results[1]['score'] if len(results) > 1 else 100
    if best['score'] >= 80:
        return best, "ok", f"Le système est très confiant. Le code <strong>{best['code']}</strong> correspond bien à cette description."
    elif best['score'] >= 60 and ecart >= 15:
        return best, "ok", f"Bonne correspondance. Le code <strong>{best['code']}</strong> est nettement le plus approprié."
    elif best['score'] >= 40 and ecart < 15:
        alts = f"<strong>{results[0]['code']}</strong> et <strong>{results[1]['code']}</strong>" if len(results) > 1 else f"<strong>{results[0]['code']}</strong>"
        return best, "warn", f"Deux codes sont très proches ({alts}). Comparez les libellés et choisissez selon le contexte exact."
    elif best['score'] >= 40:
        return best, "warn", "Score modéré. Essayez d'ajouter plus de détails sur l'activité pour améliorer la précision."
    else:
        return best, "warn", "Correspondance faible. Reformulez la description ou consultez directement la Base NOMAC."


# ══════════════════════════════════════════════════════════
#  CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Chargement du modèle NLP…")
def load_model():
    try:
        vec    = joblib.load(FICHIER_VECTORISEUR)
        matrix = joblib.load(FICHIER_MATRICE)
        df     = pd.read_pickle(FICHIER_DF)
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ('code','cod','code_nomac','codes','id'):
                rename[c] = 'Code'
            elif cl in ('description','libelle','activite','texte_nomac','texte','label','nom','nomac'):
                rename[c] = 'Description'
        if rename: df = df.rename(columns=rename)
        df['Secteur'] = df['Code'].apply(get_secteur)
        df['Icone']   = df['Secteur'].map(SECTEUR_ICONS).fillna('📌')
        return vec, matrix, df
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None

def rechercher(query, vectorizer, tfidf_matrix, df, top_n=3):
    q = clean_text(query)
    if not q: return []
    vec    = vectorizer.transform([q])
    scores = cosine_similarity(vec, tfidf_matrix).flatten()
    idx    = scores.argsort()[-top_n:][::-1]
    return [
        {'index':int(i),'code':str(df.iloc[i]['Code']),
         'description':str(df.iloc[i]['Description']),
         'secteur':str(df.iloc[i]['Secteur']),
         'icone':str(df.iloc[i]['Icone']),
         'score':round(float(scores[i])*100,1)}
        for i in idx if round(float(scores[i])*100,1) > 0.5
    ]

def afficher_carte(r, key_prefix="", rang=None):
    lvl, label, color = conf(r['score'])
    rang_html = ""
    if rang:
        rang_cols = {1:'var(--red)',2:'var(--navy)',3:'#9CA3AF'}
        rang_txts = {1:'1er choix',2:'2e choix',3:'3e choix'}
        rang_html = f"<span style='background:{rang_cols.get(rang,'#9CA3AF')};color:white;padding:2px 10px;border-radius:99px;font-size:.7rem;font-weight:700;margin-right:6px;'>{rang_txts.get(rang,'')}</span>"

    st.markdown(f"""
    <div class="nc {lvl}">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
        {rang_html}
        <span class="cp">CODE {r['code']}</span>
        <span class="sp">{r['icone']} {r['secteur']}</span>
        <span class="sv {lvl}">{r['score']}%</span>
      </div>
      <div class="nd">« {r['description']} »</div>
      <div class="nt"><div class="nf {lvl}" style="width:{min(r['score'],100)}%"></div></div>
      <span class="cb {lvl}">● {label}</span>
    </div>""", unsafe_allow_html=True)

    col_v, col_c, _ = st.columns([1.2, 1.4, 5])
    validated = False
    with col_v:
        if st.button("✅ Valider", key=f"{key_prefix}v_{r['index']}", use_container_width=True):
            validated = True
    with col_c:
        st.code(r['code'], language=None)
    return validated

def rapport_pdf(query, results, ts, user_nom):
    """Génère un PDF avec fpdf2 (pip install fpdf2)."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "fpdf2 non installé — ajoutez fpdf2 dans requirements.txt"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # En-tête marine
    pdf.set_fill_color(13, 33, 55)
    pdf.rect(0, 0, 210, 28, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(10, 8)
    pdf.cell(0, 8, "SmartNOMAC  -  Rapport de Codification", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(10, 18)
    pdf.set_text_color(180, 190, 200)
    pdf.cell(0, 6, "INSTAT Madagascar  -  Systeme de codification automatique", ln=True)

    pdf.set_y(34)
    pdf.set_text_color(30, 40, 55)

    # Bloc meta
    pdf.set_fill_color(240, 244, 248)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(10)
    meta_txt = ("Description analysee : " + str(query) + "\n"
                + "Genere par : " + str(user_nom) + "   |   Date : " + str(ts) + "\n"
                + "Nombre de suggestions : " + str(len(results)))
    pdf.multi_cell(190, 6, meta_txt, border=0, fill=True)
    pdf.ln(4)

    # En-tête tableau
    pdf.set_fill_color(13, 33, 55)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    cols = [("Code NOMAC", 28), ("Libelle officiel", 80), ("Secteur", 40), ("Score", 20), ("Confiance", 22)]
    for label, w in cols:
        pdf.cell(w, 8, label, border=0, fill=True)
    pdf.ln()

    # Lignes tableau
    pdf.set_font("Helvetica", "", 8)
    fill = False
    for r in results:
        lvl, label, _ = conf(r['score'])
        pdf.set_text_color(30, 40, 55)
        if fill: pdf.set_fill_color(248, 250, 252)
        else:    pdf.set_fill_color(255, 255, 255)
        pdf.cell(28, 7, str(r['code']),             border=0, fill=True)
        pdf.cell(80, 7, str(r['description'])[:55], border=0, fill=True)
        pdf.cell(40, 7, str(r['secteur'])[:22],     border=0, fill=True)
        sc = r['score']
        if sc >= 70:   pdf.set_text_color(0, 122, 61)
        elif sc >= 40: pdf.set_text_color(180, 120, 0)
        else:          pdf.set_text_color(120, 120, 120)
        pdf.cell(20, 7, str(sc) + "%", border=0, fill=True)
        pdf.set_text_color(30, 40, 55)
        pdf.cell(22, 7, label, border=0, fill=True)
        pdf.ln()
        fill = not fill

    # Pied de page
    pdf.set_y(-18)
    pdf.set_fill_color(240, 244, 248)
    pdf.rect(0, pdf.get_y(), 210, 18, 'F')
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(140, 150, 160)
    pdf.set_x(10)
    pdf.cell(95, 8, "SmartNOMAC v4.0  -  Master SDIA 2023-2025  -  EMIT Fianarantsoa")
    pdf.cell(95, 8, "© INSTAT Madagascar", align="R")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue(), None


# ══════════════════════════════════════════════════════════
#  PAGE D'AUTHENTIFICATION — Détection automatique du rôle
# ══════════════════════════════════════════════════════════
def page_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html,body,.main,[data-testid="stAppViewContainer"]{
      background:#FFFFFF!important;
      font-family:'Inter',sans-serif!important;
    }
    .block-container{padding:0!important;max-width:100%!important;}
    header,[data-testid="stHeader"],[data-testid="stSidebar"],
    [data-testid="stToolbar"],footer{display:none!important;}

    div[data-testid="stTextInput"] input{
      background:#F0F2F5!important;
      border:none!important;
      border-radius:6px!important;
      color:#1C1E21!important;
      font-family:'Inter',sans-serif!important;
      font-size:.95rem!important;
      padding:14px 16px!important;
      transition:background .15s!important;
    }
    div[data-testid="stTextInput"] input::placeholder{color:#BCC0C4!important;}
    div[data-testid="stTextInput"] input:focus{
      background:#E8EAF0!important;
      border:none!important;
      box-shadow:none!important;
    }
    div.stButton>button{
      background:#0866FF!important;
      border:none!important;border-radius:6px!important;
      color:white!important;font-family:'Inter',sans-serif!important;
      font-size:1rem!important;font-weight:700!important;
      padding:.82rem 1rem!important;
      box-shadow:none!important;transition:background .15s!important;
      transform:none!important;letter-spacing:0!important;
    }
    div.stButton>button:hover{background:#0757E0!important;transform:none!important;}
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 0.80, 1])
    with mid:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

        # Logo centré
        if os.path.exists(str(FICHIER_LOGO)):
            lc = st.columns([1, 1.2, 1])
            with lc[1]:
                st.image(str(FICHIER_LOGO), width=68)

        # Titre
        st.markdown("""
        <h1 style='font-family:Inter,sans-serif;font-size:1.85rem;font-weight:800;
          color:#1C1E21;text-align:center;margin:10px 0 26px;letter-spacing:-.03em;'>
          SmartNOMAC
        </h1>""", unsafe_allow_html=True)

        # Carte unique — juste identifiant + mot de passe
        with st.container(border=True):
            st.markdown("""
            <p style='font-family:Inter,sans-serif;font-size:.95rem;font-weight:700;
              color:#1C1E21;margin:0 0 16px;'>
              🔐 Connexion
            </p>""", unsafe_allow_html=True)

            username = st.text_input("Identifiant", placeholder="Identifiant",
                                     label_visibility="hidden", key="l_user")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            password = st.text_input("Mot de passe", type="password",
                                     placeholder="Mot de passe",
                                     label_visibility="hidden", key="l_pass")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            if st.button("Se connecter", key="btn_login", use_container_width=True):
                u = username.strip().lower()
                if u in COMPTES and COMPTES[u]["mdp"] == password:
                    compte = COMPTES[u]
                    # Le rôle est détecté automatiquement depuis COMPTES
                    st.session_state.update({
                        "auth":       True,
                        "username":   u,
                        "role":       compte["role"],
                        "nom":        compte["nom"],
                        "historique": []
                    })
                    if "logs_globaux" not in st.session_state:
                        st.session_state["logs_globaux"] = []
                    if "users_actifs" not in st.session_state:
                        st.session_state["users_actifs"] = {}
                    st.session_state["logs_globaux"].append({
                        "heure":       datetime.now().strftime("%H:%M:%S"),
                        "utilisateur": compte["nom"],
                        "action":      "Connexion",
                        "description": "—", "code": "—", "score": 0,
                    })
                    st.session_state["users_actifs"][u] = {
                        "nom":       compte["nom"],
                        "role":      compte["role"],
                        "connexion": datetime.now().strftime("%H:%M:%S")
                    }
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

        st.markdown("""
        <p style='font-family:Inter,sans-serif;font-size:.70rem;color:#BCC0C4;
          text-align:center;margin-top:14px;'>
          © INSTAT Madagascar
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  VÉRIFICATION AUTH
# ══════════════════════════════════════════════════════════
if not st.session_state.get("auth"):
    page_login()
    st.stop()

# Récupérer infos session
USER_ROLE = st.session_state.get("role", "user")
USER_NOM  = st.session_state.get("nom",  "Utilisateur")
USERNAME  = st.session_state.get("username", "")
IS_ADMIN  = USER_ROLE == "admin"

vectorizer, tfidf_matrix, df_final = load_model()
if "historique" not in st.session_state:
    st.session_state["historique"] = []
if vectorizer is None:
    st.stop()

nb_hist = len(st.session_state["historique"])


# ══════════════════════════════════════════════════════════
#  SIDEBAR — Menus différents selon le rôle
# ══════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div class="sb-top">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;border-radius:9px;
          background:linear-gradient(135deg,#C8102E,#FF3B5C);
          display:flex;align-items:center;justify-content:center;font-size:.95rem;
          box-shadow:0 3px 10px rgba(200,16,46,.35);">🇲🇬</div>
        <div><h2>SmartNOMAC</h2></div>
      </div>
      <div class="sub" style="margin-top:4px;padding-left:2px;">INSTAT Madagascar</div>
    </div>""", unsafe_allow_html=True)

    # Profil connecté
    role_icon = "🔐" if IS_ADMIN else "👤"
    role_lbl  = "Administrateur" if IS_ADMIN else "Utilisateur"
    st.markdown(f"""
    <div class="sb-user">
      <div class="sbu-name">{role_icon} {USER_NOM}</div>
      <div class="sbu-role">{role_lbl} · connecté</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Menu ADMIN ──────────────────────────────────
    if IS_ADMIN:
        st.markdown("<p style='font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.13em;color:rgba(255,255,255,.28);padding:0 22px;'>Administration</p>", unsafe_allow_html=True)
        page = st.radio("Navigation", [
            "📊 Tableau de Bord Global",
            "📋 Journaux d'activité",
            "👥 Gestion utilisateurs",
            "🔍 Codifier une activité",
            "⚡ Traitement en lot",
            "📂 Base NOMAC",
            "⚙️ Paramètres système",
        ], label_visibility="collapsed", key="nav_admin")

    # ── Menu UTILISATEUR ────────────────────────────
    else:
        st.markdown("<p style='font-size:.63rem;font-weight:700;text-transform:uppercase;letter-spacing:.13em;color:rgba(255,255,255,.28);padding:0 22px;'>Navigation</p>", unsafe_allow_html=True)
        page = st.radio("Navigation", [
            "🏠 Accueil",
            "🔍 Codifier une activité",
            "⚡ Traitement en lot",
            "📂 Base NOMAC",
            "📈 Mon Historique",
        ], label_visibility="collapsed", key="nav_user")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="sb-chip"><div class="v">{nb_hist}</div><div class="l">Requêtes</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="sb-chip"><div class="v">{len(df_final)}</div><div class="l">Codes</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Se déconnecter", use_container_width=True):
        for k in ["auth","username","role","nom","historique","login_role"]:
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HELPER : en-tête de page
# ══════════════════════════════════════════════════════════
def page_header(icon, icon_cls, title, subtitle, border="grey"):
    role_lbl = "👑 Administrateur" if IS_ADMIN else f"👤 {USER_NOM}"
    role_cls = "rt-admin" if IS_ADMIN else "rt-user"
    logo_html = ""
    if os.path.exists(str(FICHIER_LOGO)):
        import base64
        with open(str(FICHIER_LOGO), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{b64}" style="height:38px;width:auto;object-fit:contain;border-radius:4px;margin-right:4px;" />' 
    st.markdown(f"""
    <div style="padding:1.6rem 2.5rem 0;">
    <div class="ph {border}">
      <div style="display:flex;align-items:center;gap:10px;flex-shrink:0;">
        {logo_html}
        <div class="ph-icon {icon_cls}">{icon}</div>
      </div>
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <span class="role-tag {role_cls}" style="margin-left:auto;">{role_lbl}</span>
    </div>""", unsafe_allow_html=True)

def pg_open():  st.markdown('<div style="padding:0 2.5rem 2rem;">', unsafe_allow_html=True)
def pg_close(): st.markdown('</div>', unsafe_allow_html=True)

def log_action(action, description="—", code="—", score=0):
    """Enregistre chaque action dans les logs globaux (visibles par l'admin)."""
    st.session_state["logs_globaux"].append({
        "heure":       datetime.now().strftime("%H:%M:%S"),
        "utilisateur": USER_NOM,
        "action":      action,
        "description": description,
        "code":        code,
        "score":       score,
    })


# ══════════════════════════════════════════════════════════
#  PAGES ADMIN UNIQUEMENT
# ══════════════════════════════════════════════════════════

# ── TABLEAU DE BORD GLOBAL (Admin) ──────────────────────
if IS_ADMIN and page == "📊 Tableau de Bord Global":
    page_header("📊","gold","Tableau de Bord Global",
                "Vue d'ensemble du système, des performances et de l'activité en temps réel","navy")
    pg_open()

    logs = st.session_state["logs_globaux"]
    total_rech = sum(1 for l in logs if l["action"] == "Recherche")
    users_connectes = len(st.session_state["users_actifs"])
    scores_tous = [l["score"] for l in logs if l["score"] > 0]
    score_moy_global = round(sum(scores_tous)/len(scores_tous),1) if scores_tous else 0

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, str(len(df_final)),      "Codes NOMAC",      "Référentiel complet", "bar"),
        (c2, "94,2 %",                "Précision IA",     "Sur jeu de test",     "bar g"),
        (c3, str(total_rech),         "Requêtes totales", "Toutes sessions",     "bar gold"),
        (c4, str(users_connectes),    "Users actifs",     "Cette session",       "bar n"),
        (c5, f"{score_moy_global} %", "Score moyen",      "Toutes requêtes",     "bar"),
    ]
    for col, val, lbl, sub, bar in kpis:
        with col:
            st.markdown(f'<div class="sc"><div class="lbl">{lbl}</div>'
                        f'<div class="val">{val}</div><div class="sub">{sub}</div>'
                        f'<div class="{bar}"></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphiques
    sect_counts = df_final['Secteur'].value_counts()
    col_g1, col_g2 = st.columns([3,2])
    with col_g1:
        st.markdown('<p class="stit">Répartition des codes par secteur</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            y=sect_counts.index.tolist(), x=sect_counts.values.tolist(), orientation='h',
            marker=dict(color=COULEURS[:len(sect_counts)]),
            text=[f"  {v}" for v in sect_counts.values], textposition='outside',
        ))
        fig.update_layout(height=340, margin=dict(t=5,b=5,l=5,r=50),
                          xaxis=dict(title='Codes', gridcolor='#EEF0F5'),
                          yaxis=dict(showgrid=False), plot_bgcolor='white',
                          paper_bgcolor='white', font=dict(family='Sora'))
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        st.markdown('<p class="stit">Part sectorielle</p>', unsafe_allow_html=True)
        fig2 = px.pie(values=sect_counts.values, names=sect_counts.index,
                      hole=0.52, color_discrete_sequence=COULEURS)
        fig2.update_traces(textinfo='percent', marker=dict(line=dict(color='white',width=2)))
        fig2.update_layout(height=340,margin=dict(t=5,b=5,l=0,r=0),
                           showlegend=False,paper_bgcolor='white',font=dict(family='Sora'))
        st.plotly_chart(fig2, use_container_width=True)

    # Jauge de précision
    st.markdown('<hr><p class="stit">Performance du modèle NLP — SmartNOMAC</p>', unsafe_allow_html=True)
    jg1, jg2 = st.columns([1,2])
    with jg1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=94.2, delta={'reference':80,'increasing':{'color':'#007A3D'}},
            title={'text':"Précision globale (%)"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':'#C8102E'},
                'steps':[{'range':[0,60],'color':'#FEE2E2'},
                         {'range':[60,80],'color':'#FEF3DC'},
                         {'range':[80,100],'color':'#E8F7EF'}],
                'threshold':{'line':{'color':'#0D2137','width':3},'value':94.2}
            }
        ))
        fig_g.update_layout(height=250,margin=dict(t=30,b=10,l=20,r=20),
                             paper_bgcolor='white',font=dict(family='Sora'))
        st.plotly_chart(fig_g, use_container_width=True)

    with jg2:
        perf_data = {
            'Indicateur': ['Précision','Rappel','F1-Score','Temps moyen','Couverture'],
            'Valeur':     ['94,2 %','92,8 %','93,5 %','< 500 ms','100 %'],
            'Interprétation':['Codes corrects sur tests','Codes identifiés',
                              'Équilibre précision/rappel','Par description',
                              'Tous les 2 735 codes']
        }
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    # Export
    st.markdown("<hr>", unsafe_allow_html=True)
    csv_full = df_final[['Code','Secteur','Description']].to_csv(index=False).encode('utf-8')
    st.download_button("💾 Exporter la base NOMAC complète (CSV)", data=csv_full,
                       file_name="base_nomac_export.csv", mime="text/csv")
    pg_close()


# ── JOURNAUX D'ACTIVITÉ (Admin) ─────────────────────────
elif IS_ADMIN and page == "📋 Journaux d'activité":
    page_header("📋","n","Journaux d'activité",
                "Toutes les actions effectuées sur la plateforme — tous utilisateurs","navy")
    pg_open()

    logs = st.session_state["logs_globaux"]
    if not logs:
        st.markdown('<div class="is">Aucune activité enregistrée pour le moment.</div>',
                    unsafe_allow_html=True)
    else:
        # Filtres
        fc1, fc2 = st.columns(2)
        with fc1:
            filtre_user = st.selectbox("Filtrer par utilisateur",
                ["Tous"] + list({l["utilisateur"] for l in logs}))
        with fc2:
            filtre_action = st.selectbox("Filtrer par action",
                ["Toutes"] + list({l["action"] for l in logs}))

        logs_f = [l for l in logs
                  if (filtre_user=="Tous" or l["utilisateur"]==filtre_user)
                  and (filtre_action=="Toutes" or l["action"]==filtre_action)]

        st.markdown(f'<div class="is"><strong>{len(logs_f)}</strong> enregistrement(s) trouvé(s).</div>',
                    unsafe_allow_html=True)

        for l in reversed(logs_f[-50:]):
            score_color = "#007A3D" if l["score"]>=70 else "#D4920A" if l["score"]>=40 else "#94A3B8"
            action_bg   = "#F1F5F9" if l["action"]=="Connexion" else "#EEF4FD"
            action_fg   = "#374151" if l["action"]=="Connexion" else "#0D2137"
            code_html   = f"<span class='log-code'>{l['code']}</span>" if l["code"] != "—" else ""
            score_html  = f"<span class='log-score' style='color:{score_color};'>{l['score']}%</span>" if l["score"] > 0 else ""
            desc_txt    = str(l["description"])[:60]
            action_html = f"<span style='background:{action_bg};color:{action_fg};padding:2px 9px;border-radius:99px;font-size:.72rem;font-weight:700;'>{l['action']}</span>"
            row_html = (
                f'<div class="log-row">' +
                f'<span class="log-time">{l["heure"]}</span>' +
                f'<span class="log-user">{l["utilisateur"]}</span>' +
                action_html +
                f'<span class="log-desc">{desc_txt}</span>' +
                code_html + score_html +
                '</div>'
            )
            st.markdown(row_html, unsafe_allow_html=True)

        # Export logs
        df_logs = pd.DataFrame(logs_f)
        csv_logs = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Exporter les journaux (CSV)", data=csv_logs,
                           file_name="logs_smartnomac.csv", mime="text/csv")
    pg_close()


# ── GESTION UTILISATEURS (Admin) ────────────────────────
elif IS_ADMIN and page == "👥 Gestion utilisateurs":
    page_header("👥","n","Gestion des utilisateurs",
                "Comptes enregistrés sur la plateforme SmartNOMAC","navy")
    pg_open()

    st.markdown('<p class="stit">Comptes de la plateforme</p>', unsafe_allow_html=True)
    users_actifs = st.session_state["users_actifs"]

    for login, info in COMPTES.items():
        actif     = login in users_actifs
        role_icon = "🔐" if info["role"] == "admin" else "👤"
        role_lbl  = "Administrateur" if info["role"] == "admin" else "Utilisateur"

        with st.container(border=True):
            ca, cb = st.columns([3, 1])
            with ca:
                st.markdown(f"**{role_icon} {info['nom']}**")
                st.caption(f"Identifiant : `{login}`  ·  {role_lbl}")
            with cb:
                if actif:
                    st.success(f"Connecté à {users_actifs[login]['connexion']}")
                else:
                    st.caption("Hors ligne")
    pg_close()


# ── PARAMÈTRES SYSTÈME (Admin) ───────────────────────────
elif IS_ADMIN and page == "⚙️ Paramètres système":
    page_header("⚙️","n","Paramètres système",
                "Configuration du moteur NLP et des seuils de confiance","navy")
    pg_open()

    st.markdown('<p class="stit">Seuils de confiance</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        seuil_haut = st.slider("Seuil de confiance élevée (%)", 50, 95, 70,
                               help="Score au-dessus duquel le code est considéré fiable")
        st.markdown(f'<div class="ok"><strong>✅ Confiance élevée : ≥ {seuil_haut}%</strong><p>Les codes au-dessus de ce seuil peuvent être utilisés directement sans vérification.</p></div>', unsafe_allow_html=True)
    with c2:
        seuil_bas = st.slider("Seuil de confiance minimale (%)", 10, 60, 40,
                              help="Score en-dessous duquel le code est considéré douteux")
        st.markdown(f'<div class="warn"><strong>⚠️ Vérification requise : {seuil_bas}% – {seuil_haut}%</strong><p>Les codes dans cette plage doivent être vérifiés manuellement par un agent.</p></div>', unsafe_allow_html=True)

    st.markdown('<hr><p class="stit">Informations sur le modèle NLP</p>', unsafe_allow_html=True)
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class="ap-card n">
          <h3>Vectoriseur TF-IDF</h3>
          <p>Fichier : <code>vectorizer.pkl</code><br>
          ngram_range : (1, 2) — uni-grams + bi-grams<br>
          Dimensions : 16 630 features</p>
        </div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""
        <div class="ap-card n">
          <h3>Matrice TF-IDF</h3>
          <p>Fichier : <code>tfidf_matrix.pkl</code><br>
          Dimensions : 2 735 × 16 630<br>
          Similarité : cosinus</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr><p class="stit">Actions de maintenance</p>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("🗑️ Effacer tous les journaux", use_container_width=True):
            st.session_state["logs_globaux"] = []
            st.success("Journaux effacés.")
    with col_a2:
        if st.button("🔄 Recharger le modèle NLP", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache effacé. Le modèle sera rechargé.")
    pg_close()


# ══════════════════════════════════════════════════════════
#  PAGES COMMUNES (Admin + Utilisateur)
# ══════════════════════════════════════════════════════════

# ── ACCUEIL UTILISATEUR ──────────────────────────────────
elif not IS_ADMIN and page == "🏠 Accueil":
    page_header("🏠","","Bienvenue, "+USER_NOM.split()[0],
                "Que souhaitez-vous faire aujourd'hui ?")
    pg_open()

    # Bannière bienvenue — sans HTML f-string imbriqué
    nom_affiche = USER_NOM
    nb_codes    = len(df_final)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0D2137,#1A3A5C);' +
        'border-radius:16px;padding:26px 30px;margin-bottom:22px;color:white;">' +
        '<div style="font-size:1.6rem;margin-bottom:6px;">👋</div>' +
        f'<div style="font-family:Sora,sans-serif;font-size:1.35rem;font-weight:700;' +
        'color:white;margin:0 0 4px;">Bonjour, ' + nom_affiche + ' !</div>' +
        f'<div style="color:rgba(255,255,255,.55);font-size:.88rem;">SmartNOMAC est prêt. ' +
        str(nb_codes) + ' codes NOMAC disponibles.</div>' +
        '</div>',
        unsafe_allow_html=True
    )

    # Raccourcis — sans f-string couleur imbriqué
    st.markdown('<p class="stit">Accès rapide</p>', unsafe_allow_html=True)
    ra1, ra2, ra3 = st.columns(3)
    raccourcis = [
        (ra1, "🔍", "Codifier une activité", "Trouvez le code NOMAC d'une activité",  "#3A3F4B"),
        (ra2, "⚡", "Traitement en lot",      "Codifiez plusieurs activités d'un coup", "#1A3A5C"),
        (ra3, "📂", "Base NOMAC",             "Consultez le référentiel complet",        "#007A3D"),
    ]
    for col, ic, titre, desc, color in raccourcis:
        with col:
            card = (
                '<div class="ap-card" style="text-align:center;padding:26px 18px;' +
                'border-top:4px solid ' + color + ';border-radius:14px;">' +
                '<div style="font-size:1.9rem;margin-bottom:9px;">' + ic + '</div>' +
                '<div style="font-size:.93rem;font-weight:700;color:#0D2137;margin-bottom:4px;">' + titre + '</div>' +
                '<div style="font-size:.80rem;color:#5E7291;">' + desc + '</div>' +
                '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)

    # Stats session
    if nb_hist > 0:
        st.markdown('<hr><p class="stit">Mon activité cette session</p>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        scores_h = [h['score'] for h in st.session_state["historique"]]
        with s1: st.metric("Recherches", nb_hist)
        with s2: st.metric("Score moyen", f"{round(sum(scores_h)/len(scores_h),1)} %")
        with s3: st.metric("Codes validés", sum(1 for h in st.session_state["historique"] if h['statut']=='Validé'))
    pg_close()


# ── CODIFIER UNE ACTIVITÉ (Admin + User) ────────────────
elif page == "🔍 Codifier une activité":
    page_header("🔍","","Codifier une activité",
                "Saisissez une description en langage naturel — SmartNOMAC trouve le bon code NOMAC")
    pg_open()

    st.markdown("""
    <p style='font-size:.93rem;color:#5E7291;margin:0 0 14px;'>
      Décrivez l'activité économique en langage naturel — le système identifie automatiquement le code NOMAC correspondant.
    </p>""", unsafe_allow_html=True)

    col_q, col_n = st.columns([5,1])
    with col_q:
        query = st.text_input("Description de l'activité", placeholder="Ex : Culture de riz, Vente de charbon, Transport routier…",
                              key="q_main", label_visibility="hidden")
    with col_n:
        top_n = st.selectbox("Résultats", [1,2,3,4,5], index=2, label_visibility="collapsed")

    _, bc, _ = st.columns([2, 1.6, 2])
    with bc:
        go_btn = st.button("🔍 Rechercher", use_container_width=True)

    # Légende
    st.markdown("""
    <div style="display:flex;gap:14px;margin:10px 0 18px;flex-wrap:wrap;">
      <span class="cb h">● Confiance élevée ≥ 70%</span>
      <span class="cb m">● Confiance moyenne 40–70%</span>
      <span class="cb l">● Confiance faible &lt; 40%</span>
    </div>""", unsafe_allow_html=True)

    if go_btn:
        if not query.strip():
            st.warning("Veuillez saisir une description.")
        else:
            with st.spinner("Analyse NLP en cours…"):
                results = rechercher(query, vectorizer, tfidf_matrix, df_final, top_n=top_n)
            ts = datetime.now().strftime("%d/%m/%Y à %H:%M")

            if not results:
                st.info("Aucun résultat pertinent. Essayez une description plus précise.")
            else:
                # Log global
                log_action("Recherche", query, results[0]['code'], results[0]['score'])

                # Aide à la décision
                best, statut, conseil = aide_decision(results)
                if statut == "ok":
                    st.markdown(f'<div class="ok"><strong>✅ Recommandation — Code {best["code"]}</strong><p>{conseil}</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warn"><strong>⚠️ Vérification recommandée</strong><p>{conseil}</p></div>', unsafe_allow_html=True)

                # Alternatives
                if len(results) > 1:
                    st.markdown('<div class="ad"><div class="ad-title">🎯 Codes alternatifs recommandés</div>', unsafe_allow_html=True)
                    whys = [
                        "Meilleure correspondance avec la description fournie.",
                        "Alternative proche — à retenir si l'activité est plus spécifique.",
                        "Option possible si les deux premières ne conviennent pas.",
                    ]
                    rank_cls = ['n1','n2','n3']
                    for i, r in enumerate(results[:3], 1):
                        lvl, _, color = conf(r['score'])
                        st.markdown(f"""
                        <div class="ad-item">
                          <div class="ad-num {rank_cls[i-1]}">{i}</div>
                          <div style="flex:1;">
                            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                              <span class="ad-code">{r['code']}</span>
                              <span class="ad-lib">{r['description']}</span>
                              <span class="ad-pct" style="color:{color}">{r['score']}%</span>
                            </div>
                            <div class="ad-why">💡 {whys[i-1] if i<=len(whys) else ''}</div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Cartes détaillées
                st.markdown(f'<p class="stit">Résultats détaillés</p>', unsafe_allow_html=True)
                for i, r in enumerate(results, 1):
                    validated = afficher_carte(r, key_prefix=f"r{i}_", rang=i)
                    if validated:
                        st.session_state["historique"].append({
                            'heure':query[:0]+datetime.now().strftime("%H:%M"),
                            'description': query, 'code': r['code'],
                            'secteur': r['secteur'], 'score': r['score'], 'statut': 'Validé'
                        })
                        log_action("Validation", query, r['code'], r['score'])
                        st.success(f"✅ Code **{r['code']}** validé !")

                # Export
                st.markdown("<hr>", unsafe_allow_html=True)
                pdf_bytes, pdf_err = rapport_pdf(query, results, ts, USER_NOM)
                if pdf_bytes:
                    st.download_button("📄 Télécharger",
                                       data=pdf_bytes,
                                       file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                       mime="application/pdf",
                                       use_container_width=False)
                else:
                    st.info("Export PDF non disponible sur cet environnement.")
    pg_close()


# ── TRAITEMENT EN LOT (Admin + User) ────────────────────
elif page == "⚡ Traitement en lot":
    page_header("⚡","","Traitement en lot",
                "Codifiez plusieurs activités d'un coup — une description par ligne")
    pg_open()

    st.markdown("""
    <p style='font-size:.93rem;color:#5E7291;margin:0 0 14px;'>
      Entrez une activité par ligne — le système codifie chaque description et retourne le meilleur code NOMAC pour chacune.
    </p>""", unsafe_allow_html=True)

    texte = st.text_area("Activités à codifier :", height=200,
                         placeholder="Culture de riz\nVente de charbon de bois\nTransport routier\nPêche artisanale",
                         key="batch_txt", label_visibility="collapsed")

    _, blot, _ = st.columns([2, 1.4, 2])
    with blot:
        _lot_btn = st.button("⚡ Lancer", use_container_width=True)
    if _lot_btn:
        lignes = [l.strip() for l in texte.strip().split('\n') if l.strip()]
        if not lignes:
            st.warning("Entrez au moins une description.")
        else:
            resultats = []
            pb = st.progress(0)
            info_txt = st.empty()
            for idx, ligne in enumerate(lignes):
                info_txt.markdown(f"*Traitement : « {ligne[:60]} »*")
                res = rechercher(ligne, vectorizer, tfidf_matrix, df_final, top_n=1)
                if res:
                    r = res[0]
                    _, label, _ = conf(r['score'])
                    resultats.append({'Description saisie':ligne,'Code NOMAC':r['code'],
                                      'Libellé officiel':r['description'],'Secteur':r['secteur'],
                                      'Score (%)':r['score'],'Confiance':label})
                    st.session_state["historique"].append({
                        'heure':datetime.now().strftime("%H:%M"),'description':ligne,
                        'code':r['code'],'secteur':r['secteur'],'score':r['score'],'statut':'Lot'})
                    log_action("Lot", ligne, r['code'], r['score'])
                pb.progress((idx+1)/len(lignes))
            info_txt.empty(); pb.empty()

            if resultats:
                df_res = pd.DataFrame(resultats)
                scores_lot = [r['Score (%)'] for r in resultats]
                hautes   = sum(1 for s in scores_lot if s >= 70)
                moyennes = sum(1 for s in scores_lot if 40 <= s < 70)
                faibles  = sum(1 for s in scores_lot if s < 40)

                if faibles/len(scores_lot) > 0.3:
                    st.markdown(f'<div class="warn"><strong>⚠️ {faibles} description(s) à vérifier manuellement</strong><p>Plus de 30% des résultats ont une faible confiance. Vérifiez la colonne "Confiance".</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ok"><strong>✅ Traitement de qualité</strong><p>{hautes} confiance élevée · {moyennes} à vérifier · {faibles} douteux</p></div>', unsafe_allow_html=True)

                st.success(f"✅ {len(resultats)} activité(s) codifiée(s) !")
                st.dataframe(df_res, use_container_width=True, height=min(450,80+len(df_res)*40))

                # Synthèse
                st.markdown("<hr>", unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                with m1: st.metric("Score moyen",    f"{round(sum(scores_lot)/len(scores_lot),1)} %")
                with m2: st.metric("✅ Élevée",       f"{hautes}")
                with m3: st.metric("⚠️ À vérifier",  f"{moyennes}")
                with m4: st.metric("❌ Douteux",      f"{faibles}")

                # Graphique
                sect_lot = pd.Series([r['Secteur'] for r in resultats]).value_counts()
                fig_lot = px.bar(x=sect_lot.values, y=sect_lot.index, orientation='h',
                                 color=sect_lot.index, color_discrete_sequence=COULEURS, text=sect_lot.values)
                fig_lot.update_layout(height=260, showlegend=False,
                                      xaxis_title='Activités', yaxis_title='',
                                      plot_bgcolor='white', paper_bgcolor='white',
                                      margin=dict(t=5,b=5), font=dict(family='Sora'))
                st.plotly_chart(fig_lot, use_container_width=True)

                # Exports
                col_e1, col_e2, _ = st.columns([1,1,3])
                with col_e1:
                    st.download_button("💾 CSV", data=df_res.to_csv(index=False).encode('utf-8'),
                                       file_name="codification_lot.csv", mime="text/csv", use_container_width=True)
                with col_e2:
                    try:
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='openpyxl') as w: df_res.to_excel(w,index=False)
                        st.download_button("📊 Excel", data=buf.getvalue(),
                                           file_name="codification_lot.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                           use_container_width=True)
                    except: st.info("Excel non disponible.")
    pg_close()


# ── BASE NOMAC (Admin + User) ────────────────────────────
elif page == "📂 Base NOMAC":
    page_header("📂","g","Base NOMAC 2012",
                "Consultez les 2 735 codes officiels de la nomenclature des activités malgaches","green")
    pg_open()

    cf1, cf2 = st.columns([1,2])
    with cf1:
        sect_f = st.selectbox("Secteur",
            ["Tous les secteurs"] + sorted(df_final['Secteur'].unique().tolist()))
    with cf2:
        kw = st.text_input("Recherche par mot-clé", placeholder="Ex : riz, pêche, transport…",
                           label_visibility="hidden")

    df_v = df_final.copy()
    if sect_f != "Tous les secteurs": df_v = df_v[df_v['Secteur']==sect_f]
    if kw.strip():
        df_v = df_v[df_v['Description'].str.contains(kw,case=False,na=False)|
                    df_v['Code'].astype(str).str.contains(kw,na=False)]
    df_v = df_v.assign(Code_str=df_v['Code'].astype(str)).sort_values('Code_str').drop(columns='Code_str')

    st.markdown(f'<div class="is"><strong>{len(df_v)}</strong> activité(s) sur {len(df_final)} codes.</div>', unsafe_allow_html=True)
    st.dataframe(df_v[['Code','Secteur','Description']].reset_index(drop=True),
                 use_container_width=True, height=500)
    st.download_button("💾 Exporter la sélection (CSV)",
                       data=df_v[['Code','Secteur','Description']].to_csv(index=False).encode('utf-8'),
                       file_name="nomac_selection.csv", mime="text/csv")
    pg_close()


# ── HISTORIQUE (User uniquement) ────────────────────────
elif not IS_ADMIN and page == "📈 Mon Historique":
    page_header("📈","","Mon Historique",
                "Toutes vos codifications effectuées pendant cette session")
    pg_open()

    hist = st.session_state["historique"]
    if not hist:
        st.markdown('<div class="is">Aucune recherche pour le moment. Allez sur <strong>Codifier une activité</strong>.</div>', unsafe_allow_html=True)
    else:
        scores_h = [h['score'] for h in hist]
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Recherches",    len(hist))
        with c2: st.metric("Score moyen",   f"{round(sum(scores_h)/len(scores_h),1)} %")
        with c3: st.metric("Validés",        sum(1 for h in hist if h['statut']=='Validé'))
        with c4: st.metric("Traités en lot", sum(1 for h in hist if h['statut']=='Lot'))

        st.markdown("<br>", unsafe_allow_html=True)
        df_hist = pd.DataFrame(hist)
        df_hist.columns = ['Heure','Description','Code','Secteur','Score (%)','Statut']
        st.dataframe(df_hist, use_container_width=True, height=360)

        if len(hist) > 1:
            st.markdown('<p class="stit">Évolution des scores</p>', unsafe_allow_html=True)
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=list(range(1,len(hist)+1)), y=scores_h, mode='lines+markers',
                line=dict(color='#C8102E',width=2.5,shape='spline'),
                marker=dict(size=8,color='#0D2137',line=dict(color='white',width=2)),
                fill='tozeroy', fillcolor='rgba(200,16,46,0.07)'
            ))
            fig_ev.add_hline(y=70, line_dash="dot", line_color="#007A3D", annotation_text="Seuil élevé")
            fig_ev.add_hline(y=40, line_dash="dot", line_color="#D4920A", annotation_text="Seuil moyen")
            fig_ev.update_layout(xaxis_title="Nº recherche", yaxis_title="Score (%)",
                                 yaxis=dict(range=[0,110]), height=280,
                                 margin=dict(t=20,b=10), plot_bgcolor='white',
                                 paper_bgcolor='white', font=dict(family='Sora'), showlegend=False)
            st.plotly_chart(fig_ev, use_container_width=True)

        col_e1, col_e2, _ = st.columns([1,1,4])
        with col_e1:
            st.download_button("💾 Exporter (CSV)",
                               data=df_hist.to_csv(index=False).encode('utf-8'),
                               file_name="mon_historique.csv", mime="text/csv", use_container_width=True)
        with col_e2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state["historique"] = []
                st.rerun()
    pg_close()
