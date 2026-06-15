import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import shap
import plotly.graph_objects as go
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.stApp::before{

    content:"";

    position:fixed;

    width:350px;
    height:350px;

    border-radius:50%;

    background:
        rgba(0,255,255,0.12);

    filter:blur(120px);

    top:-100px;
    left:-100px;

    z-index:-1;
}

.stApp::after{

    content:"";

    position:fixed;

    width:300px;
    height:300px;

    border-radius:50%;

    background:
        rgba(59,130,246,0.12);

    filter:blur(120px);

    bottom:-100px;
    right:-100px;

    z-index:-1;
}

/* Glass Card */

.glass-card{
    border-left:4px solid #00ffff;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);

    border:1px solid rgba(255,255,255,0.18);

    border-radius:25px;

    padding:25px;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);

    transition:0.4s;
}

/* Hover Animation */

.glass-card:hover{
    transform:translateY(-6px);
    box-shadow:
        0 12px 40px rgba(0,255,255,0.25);
}
@keyframes floatUp{
    from{
        opacity:0;
        transform:translateY(30px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}
            @keyframes pulseGlow{

    0%{
        box-shadow:
        0 0 10px rgba(0,255,255,0.15);
    }

    50%{
        box-shadow:
        0 0 30px rgba(0,255,255,0.4);
    }

    100%{
        box-shadow:
        0 0 10px rgba(0,255,255,0.15);
    }

}

.glass-card{
    animation:
        floatUp 0.8s ease forwards,
        pulseGlow 4s infinite;
}
            .glass-card:hover{
    transform:
        translateY(-8px)
        scale(1.03);

    transition:0.4s;
}
/* Titles */
    .glass-card h1{
    color:#00ffff;
    text-shadow:
        0 0 10px #00ffff,
        0 0 20px #00ffff,
        0 0 40px #00ffff;
}

.kpi-title{

    color:white;

    text-shadow:
        0 0 10px rgba(255,255,255,0.5);

    font-weight:600;
}
            .kpi-value{

    font-size:42px;

    font-weight:700;

    color:#00ffff;

    # text-shadow:
    #     0 0 10px #00ffff,
    #     0 0 20px #00ffff,
    #     0 0 40px #00ffff;

#     animation:
#         neonPulse 2s infinite;
# }
            
#             .card1{
#     animation-delay:0.2s;
# }

# .card2{
#     animation-delay:0.4s;
# }

# .card3{
#     animation-delay:0.6s;
# }

h1,h2,h3{
    color:white;
}
            /* AI Medical Header */

.ai-header{
    position:relative;

    background:linear-gradient(
        135deg,
        rgba(0,255,255,0.15),
        rgba(59,130,246,0.15)
    );

    border:1px solid rgba(255,255,255,0.15);

    backdrop-filter:blur(20px);

    border-radius:30px;

    padding:40px;

    overflow:hidden;

    margin-bottom:30px;

    box-shadow:
        0 0 40px rgba(0,255,255,0.15);
}

/* Animated Glow */

.ai-header::before{
    content:"";

    position:absolute;

    width:300px;
    height:300px;

    background:
        radial-gradient(
            circle,
            rgba(0,255,255,0.25),
            transparent
        );

    top:-100px;
    right:-100px;

    animation:
        rotateGlow 10s linear infinite;
}

@keyframes rotateGlow{

    from{
        transform:rotate(0deg);
    }

    to{
        transform:rotate(360deg);
    }
}

.ai-title{

    font-size:52px;

    font-weight:800;

    color:#00ffff;

    text-shadow:
        0 0 10px #00ffff,
        0 0 20px #00ffff,
        0 0 40px #00ffff;
}

.ai-subtitle{

    color:white;

    font-size:20px;

    margin-top:10px;
}

.ai-badge{

    display:inline-block;

    padding:8px 18px;

    border-radius:50px;

    margin-top:15px;

    background:
        rgba(0,255,255,0.12);

    border:
        1px solid rgba(0,255,255,0.3);

    color:#00ffff;

    font-weight:600;
}
            /* =====================================
   FLOATING GLASS SIDEBAR
===================================== */

section[data-testid="stSidebar"]{

    background:
        rgba(15,23,42,0.75);

    backdrop-filter:
        blur(25px);

    -webkit-backdrop-filter:
        blur(25px);

    border-right:
        1px solid rgba(255,255,255,0.15);

    box-shadow:
        8px 0px 30px rgba(0,255,255,0.15);

}

/* Sidebar Logo Animation */

section[data-testid="stSidebar"] img{

    animation:
        floatLogo 3s ease-in-out infinite;
}

@keyframes floatLogo{

    0%{
        transform:translateY(0px);
    }

    50%{
        transform:translateY(-8px);
    }

    100%{
        transform:translateY(0px);
    }
}

/* Navigation Title */

section[data-testid="stSidebar"] h1{

    color:#00ffff;

    text-align:center;

    text-shadow:
        0 0 10px #00ffff,
        0 0 20px #00ffff;
}
            /* Menu Buttons */

.stRadio > div{

    gap:10px;
}

.stRadio label{

    background:
        rgba(255,255,255,0.06);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius:15px;

    padding:12px; 

    transition:0.3s;

    backdrop-filter:blur(10px);
}

.stRadio label:hover{

    transform:translateX(8px);

    border:
        1px solid #00ffff;

    box-shadow:
        0 0 15px rgba(0,255,255,0.3);
}
            /* =====================================
   NEON BUTTONS
===================================== */

.stButton > button{

    width:100%;

    background:
        linear-gradient(
            135deg,
            #00ffff,
            #3b82f6
        );

    color:white;

    font-size:18px;

    font-weight:700;

    border:none;

    border-radius:15px;

    padding:12px;

    transition:0.4s;

    box-shadow:
        0 0 15px rgba(0,255,255,0.4);

}

/* Hover Effect */

.stButton > button:hover{

    transform:
        translateY(-4px)
        scale(1.02);

    box-shadow:
        0 0 20px #00ffff,
        0 0 40px #00ffff,
        0 0 60px #00ffff;
}

/* Click Effect */

.stButton > button:active{

    transform:scale(0.97);
}
            /* Download Button */

.stDownloadButton > button{

    width:100%;

    background:
        linear-gradient(
            135deg,
            #10b981,
            #00ffff
        );

    color:white;

    font-weight:700;

    border:none;

    border-radius:15px;

    padding:12px;

    box-shadow:
        0 0 15px rgba(16,185,129,0.4);

    transition:0.4s;
}

.stDownloadButton > button:hover{

    box-shadow:
        0 0 20px #10b981,
        0 0 40px #10b981;

    transform:
        translateY(-4px);
}
            @keyframes neonPulse{

    0%{
        box-shadow:
            0 0 10px rgba(0,255,255,0.3);
    }

    50%{
        box-shadow:
            0 0 30px rgba(0,255,255,0.8);
    }

    100%{
        box-shadow:
            0 0 10px rgba(0,255,255,0.3);
    }
}

.stButton > button{

    animation:
        neonPulse 2s infinite;
}
            /* =====================================
   WELCOME SCREEN
===================================== */

.welcome-screen{

    padding-top:80px;
}

.welcome-title{

    font-size:58px;

    font-weight:800;

    color:#00ffff;

    text-shadow:
        0 0 10px #00ffff,
        0 0 20px #00ffff,
        0 0 40px #00ffff;

    animation:
        glowTitle 2s infinite alternate;
}

.welcome-subtitle{

    color:white;

    font-size:22px;

    margin-top:20px;
}

@keyframes glowTitle{

    from{
        text-shadow:
            0 0 10px #00ffff;
    }

    to{
        text-shadow:
            0 0 25px #00ffff,
            0 0 50px #00ffff;
    }
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

@st.cache_resource
def load_model():

    model = joblib.load(
        "breast_cancer_model.pkl"
    )

    scaler = joblib.load(
        "breast_cancer_scaler.pkl"
    )

    features = joblib.load(
        "feature_names.pkl"
    )

    return model, scaler, features
@st.cache_resource
def load_shap_explainer():

    model, scaler, features = load_model()

    explainer = shap.KernelExplainer(
        model.predict_proba,
        np.zeros((1, len(features)))
    )

    return explainer
def load_lottie_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
medical_ai_animation = load_lottie_file(
    "medical_animation.json"
)

def generate_pdf(
    result,
    confidence,
    top_features=None
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(
    Paragraph(
        "🩺 AI Breast Cancer Diagnostic Report",
        styles['Title']
    )
)
    report_id = datetime.now().strftime(
    "BCP-%Y%m%d-%H%M%S"
)

    content.append(
    Paragraph(
        f"<b>Report ID:</b> {report_id}",
        styles['Normal']
    )
)

    content.append(
    Paragraph(
        "Advanced Explainable Artificial Intelligence System",
        styles['Italic']
    )
)

    content.append(Spacer(1, 15))

    # Result
    content.append(
        Paragraph(
            f"<b>Prediction Result:</b> {result}",
            styles['Heading2']
        )
    )
    risk_level = (
    "High Risk"
    if result == "Malignant"
    else "Low Risk"
)

    content.append(
    Paragraph(
        f"<b>AI Risk Assessment:</b> {risk_level}",
        styles['Heading2']
    )
)
    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            f"<b>Model Confidence:</b> {confidence:.2f}%",
            styles['Heading2']
        )
    )
    if confidence >= 90:
        confidence_text = "Very High"

    elif confidence >= 75:
        confidence_text = "High"

    else:
        confidence_text = "Moderate"

    content.append(
        Paragraph(
            f"<b>Confidence Level:</b> {confidence_text}",
            styles['Normal']
        )
    )
    
    content.append(Spacer(1, 12))

    # Description Section

    if result == "Malignant":

        content.append(
            Paragraph(
                "<b>Result Explanation:</b>",
                styles['Heading2']
            )
        )

        content.append(
            Paragraph(
                "The machine learning model predicts that the tumor "
                "shows characteristics commonly associated with a "
                "malignant (cancerous) breast tumor.",
                styles['Normal']
            )
        )

        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                "<b>Severity Level:</b> High Risk",
                styles['Normal']
            )
        )

        content.append(
            Paragraph(
                "A malignant tumor has the potential to grow rapidly "
                "and spread to surrounding tissues or other parts of "
                "the body if left untreated.",
                styles['Normal']
            )
        )

        content.append(
            Paragraph(
                "Immediate consultation with an oncologist or "
                "qualified healthcare professional is strongly "
                "recommended.",
                styles['Normal']
            )
        )
        content.append(
         Paragraph(
        "<b>Recommended Next Steps:</b>",
        styles['Heading2']
    )
)

        content.append(
         Paragraph(
        "• Consult an oncologist immediately<br/>"
        "• Schedule additional diagnostic tests<br/>"
        "• Review pathology findings<br/>"
        "• Seek professional medical advice",
        styles['Normal']
    )
)

    else:

        content.append(
            Paragraph(
                "<b>Result Explanation:</b>",
                styles['Heading2']
            )
        )

        content.append(
            Paragraph(
                "The machine learning model predicts that the tumor "
                "shows characteristics commonly associated with a "
                "benign (non-cancerous) breast tumor.",
                styles['Normal']
            )
        )
        content.append(
    Paragraph(
        "<b>Recommended Next Steps:</b>",
        styles['Heading2']
    )
)

        content.append(
    Paragraph(
        "• Continue regular screenings<br/>"
        "• Maintain healthy lifestyle habits<br/>"
        "• Follow physician recommendations",
        styles['Normal']
    )
)
        content.append(Spacer(1, 8))

        content.append(
            Paragraph(
                "<b>Severity Level:</b> Low Risk",
                styles['Normal']
            )
        )

        content.append(
            Paragraph(
                "Benign tumors generally do not spread to other parts "
                "of the body and are less likely to pose a serious "
                "health threat.",
                styles['Normal']
            )
        )

        content.append(
            Paragraph(
                "Regular medical monitoring and consultation with a "
                "healthcare professional are still advised.",
                styles['Normal']
            )
        )
        content.append(Spacer(1, 15))

# Explainable AI Section
        content.append(
    Paragraph(
        "<b>Explainable AI:</b> "
        "SHAP analysis was performed "
        "to identify the most influential "
        "features affecting the prediction.",
        styles['Normal']
    )
)

        content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "<b>Top Influential Features:</b>",
            styles['Heading2']
        )
    )

    if top_features:

        for feature in top_features:

            content.append(
                Paragraph(
                    f"• {feature}",
                    styles['Normal']
                )
            )

    content.append(Spacer(1, 15))

    # Disclaimer

    content.append(
        Paragraph(
            "<b>Important Disclaimer</b>",
            styles['Heading2']
        )
    )

    content.append(
        Paragraph(
            "This prediction is generated using a Support Vector "
            "Machine (SVM) machine learning model trained on the "
            "Breast Cancer Wisconsin Dataset. The result should be "
            "used only for educational and screening purposes and "
            "must not replace professional medical diagnosis.",
            styles['Normal']
        )
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Generated by Breast Cancer Prediction System",
            styles['Italic']
        )
    )
    content.append(Spacer(1, 15))
    content.append(
            Paragraph(
                f"<b>Report Generated On:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
                styles['Normal']
            )
        )
        

    content.append(
        Paragraph(
            "<b>Model Information</b>",
            styles['Heading2']
        )
    )

    content.append(
        Paragraph(
            "Algorithm: Support Vector Machine (RBF Kernel)<br/>"
            "Dataset: Breast Cancer Wisconsin Dataset<br/>"
            "Features Used: 30 Clinical Features<br/>"
            "AI Module: Explainable AI",
            styles['Normal']
        )
    )    
    content.append(
        Spacer(1, 20)
    )

    content.append(
        Paragraph(
            "Generated by AI Breast Cancer Detection System",
            styles['Italic']
        )
    )

    content.append(
        Paragraph(
            "© 2026 Medical AI Research Project",
            styles['Italic']
        )
    )

    doc.build(content)

    buffer.seek(0)

    return buffer
# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=120
)

st.sidebar.markdown("""
<h2 style="
color:#00ffff;
text-align:center;
text-shadow:
0 0 10px #00ffff,
0 0 20px #00ffff;">
🩺 AI MEDICAL HUB
</h2>
""",
unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="
background:rgba(255,255,255,0.06);
padding:15px;
border-radius:15px;
border:1px solid rgba(0,255,255,0.2);
margin-top:15px;
text-align:center;">

<h4 style="color:#00ffff;">
🟢 System Online
</h4>

AI Diagnostic Engine Ready

</div>
""",
unsafe_allow_html=True)

# Permanent Dark Theme

st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

h1,h2,h3,h4,h5,h6{
    color:white;
}

section[data-testid="stSidebar"]{
    background-color:#161B22;
}

</style>
""",
unsafe_allow_html=True)

menu = st.sidebar.radio("",
    [
        "Home",
        "Dataset Overview",
        "EDA Dashboard",
        "Train Model",
        "Prediction"
    ],
    label_visibility="hidden"
)

# ------------------------------------------------
# HOME
# ------------------------------------------------

if menu == "Home":

    st.title("🩺 Breast Cancer Prediction Dashboard")
    col1, col2 = st.columns([1,1])

    with col1:

     st_lottie(
    medical_ai_animation,
    height=400,
    key="medical_ai"
)

    with col2:

      st.markdown("""
    <div class="welcome-screen">

    </div>
    """,
    unsafe_allow_html=True)

    st.markdown("""
<div class="ai-header">

<div class="ai-title">
🩺 AI Breast Cancer Detection System
</div>
<div class="ai-subtitle">
Advanced Explainable Artificial Intelligence Platform
for Early Breast Cancer Risk Assessment
</div>
<div class="ai-badge">
🧠 SVM AI Engine • Explainable AI • Real-Time Prediction
</div>

</div>
""",
unsafe_allow_html=True)
    colA,colB,colC,colD = st.columns(4)

    with colA:
      st.info("🧬 Tumor Analysis")

    with colB:
      st.info("🤖 AI Prediction")

    with colC:
      st.info("📊 Explainable AI")

    with colD:
      st.info("📄 Medical Reports")

    c1,c2,c3 = st.columns(3)

    with c1:
      st.markdown(f"""
    <div class="glass-card card1">
        <h2 class="kpi-title">📊 Dataset Rows</h2>
        <div class="kpi-value">
    {df.shape[0]}
</div>
    </div>
    """,
    unsafe_allow_html=True)

    with c2:
      st.markdown("""
    <div class="glass-card card2">
        <h2 class="kpi-title"> 🧬 Features Used</h2>
        <div class="kpi-value">
    30
</div>
    </div>
    """,
    unsafe_allow_html=True)

    with c3:
       st.markdown("""
    <div class="glass-card card3">
        <h2 class="kpi-title">🤖 Model</h2>
        <div class="kpi-value">
    SVM
</div>
    </div>
    """,
    unsafe_allow_html=True)
       st.markdown("<br>", unsafe_allow_html=True)

    
# ------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------

elif menu == "Dataset Overview":

    st.title("Dataset Overview")

    st.subheader("First 5 Rows")

    st.dataframe(df.head())

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Data Types")

    st.write(df.dtypes)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

# ------------------------------------------------
# EDA DASHBOARD
# ------------------------------------------------

elif menu == "EDA Dashboard":

    st.title("Exploratory Data Analysis")

    df_clean = df.copy()

    df_clean.drop(
        ["id","Unnamed: 32"],
        axis=1,
        inplace=True
    )

    st.subheader("Diagnosis Distribution")

    fig = px.histogram(
        df_clean,
        x="diagnosis",
        color="diagnosis"
    )

    st.plotly_chart(fig,
                    use_container_width=True)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(15,10)
    )

    sns.heatmap(
        df_clean.select_dtypes(
            include=np.number
        ).corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Feature Distribution")

    feature = st.selectbox(
        "Select Feature",
        df_clean.columns[1:]
    )

    fig2 = px.histogram(
        df_clean,
        x=feature,
        color="diagnosis"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------

elif menu == "Train Model":

    st.title("Model Training")

    data = df.copy()

    data.drop(
        ['id','Unnamed: 32'],
        axis=1,
        inplace=True
    )

    encoder = LabelEncoder()

    data['diagnosis'] = encoder.fit_transform(
        data['diagnosis']
    )

    X = data.drop(
        'diagnosis',
        axis=1
    )
    joblib.dump(X.columns.tolist(), "feature_names.pkl")

    y = data['diagnosis']

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(
    kernel='rbf',
    probability=True
)

    model.fit(
        X_train,
        y_train
    )
    joblib.dump(model, "breast_cancer_model.pkl")
    joblib.dump(scaler, "breast_cancer_scaler.pkl")

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.success(
        f"Accuracy : {accuracy*100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        ax=ax
    )
    st.pyplot(fig)

    report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df.round(2))

# ------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------

elif menu == "Prediction":

    st.title("Live Prediction")

    model, scaler, features = load_model()

    explainer = load_shap_explainer()

    user_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].mean())

        if i % 2 == 0:

            with col1:
                user_data[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )

        else:

            with col2:
                user_data[feature] = st.slider(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )

    if st.button(
    "🚀 Run AI Diagnosis",
    key="prediction_button"
    ):

        input_df = pd.DataFrame([user_data])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        probability = model.predict_proba(input_scaled)
        shap_values = explainer.shap_values(
        input_scaled
)
        # st.write("SHAP Shape:", np.array(shap_values).shape)

        if prediction[0] == 1:
            st.error("🔴 Malignant Tumor Detected")
        else:
            st.success("🟢 Benign Tumor Detected")

        confidence = max(probability[0]) * 100

        # KPI Cards
        result = (
                    "Malignant"
                    if prediction[0] == 1
                    else "Benign"
                )
        k1,k2,k3 = st.columns([1,1,1])

        with k1:
            st.markdown(f"""
    <div class="glass-card card1">
        <h3 class="kpi-title">Prediction</h3>
        <div class="kpi-value">
    {result}
    </div>
    </div>
    """,
    unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
    <div class="glass-card card2">
        <h3 class="kpi-title">Confidence</h3>
        <div class="kpi-value">
    {confidence:.2f}%
    </div>
    </div>
    """,
    unsafe_allow_html=True)

        with k3:
             st.markdown(f"""
    <div class="glass-card card3">
        <h3 class="kpi-title">Risk Level</h3>
        <div class="kpi-value">
    {"High Risk" if prediction[0]==1 else "Low Risk"}
    </div>
    </div>
    """,
    unsafe_allow_html=True)
             
        gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",

        value=round(confidence,2),

        title={
            "text":"🧠 AI Confidence Score"
        },

        gauge={
            "axis":{
                "range":[0,100]
            },

            "bar":{
                "color":"cyan"
            },

            "steps":[
                {
                    "range":[0,50],
                    "color":"#ff4d4d"
                },
                {
                    "range":[50,80],
                    "color":"#facc15"
                },
                {
                    "range":[80,100],
                    "color":"#10b981"
                }
            ],

            "threshold":{
                "line":{
                    "color":"white",
                    "width":6
                },
                "thickness":0.8,
                "value":confidence
            }
        }
    )
)

        gauge.update_layout(
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    font={"color":"white"}
)

        st.plotly_chart(
        gauge,
    use_container_width=True
)

        prob_df = pd.DataFrame({
            "Class": ["Benign", "Malignant"],
            "Probability": probability[0]
        })

        fig = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            title="Prediction Probabilities"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        st.markdown("---")
        st.subheader("🧠 Explainable AI (SHAP)")
#         fig_shap, ax = plt.subplots(figsize=(10,5))

#         shap.plots._waterfall.waterfall_legacy(
#     explainer.expected_value[0],
#     shap_values[0],
#     feature_names=features,
#     show=False
# )

#         st.pyplot(fig_shap)
        feature_importance = pd.DataFrame({
    "Feature": features,
    "Value": np.abs(input_scaled[0])
})

        feature_importance = feature_importance.sort_values(
    "Value",
    ascending=False
)
        top_features =(
                feature_importance
                .head(5)["Feature"]
                .tolist()
        )
                
        
        pdf_file = generate_pdf(
                 result,
                 confidence,
                 top_features
        )
        st.download_button(
                    label="📄 Download Prediction Report",
                    data=pdf_file,
                    file_name="Breast_Cancer_Report.pdf",
                    mime="application/pdf"
                )

        fig_bar = px.bar(
        feature_importance.head(10),
    x="Value",
    y="Feature",
    orientation="h",
    title="Top Features Influencing Prediction"
)

        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader(
    "Top Features Affecting Prediction"
)
        st.dataframe(
        feature_importance
    .sort_values("Value", ascending=False)
    .head(10)
) 
        
        st.markdown(f"""
<div class="glass-card">

<h3>🧠 AI Explanation</h3>

The model classified this tumor as
<b>{result}</b> with a confidence of
<b>{confidence:.2f}%</b>.

The top contributing features were:

<ul>
{''.join([f'<li>{f}</li>' for f in top_features])}
</ul>

</div>
""",
unsafe_allow_html=True)