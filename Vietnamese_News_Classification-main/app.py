import streamlit as st
import os

#from torch import div

st.set_page_config(
    page_title="Vietnamese News Classification Using XGBoost",
    layout="wide"
)

# ===== HEADER =====
with st.container():
    col1, col2, col3 = st.columns([1,4,1])

    with col1:
        if os.path.exists("rose.png"):
            st.image("rose.png", width=110)

    with col2:
        st.markdown(
            "<h2 style='text-align:center;'>Topic 2: Developing a Vietnamese News Classification System Using XGBoost</h2>",
            unsafe_allow_html=True
        )

    # No header box on the right
    with col3:
        pass

st.write("---")

# ===== SIDEBAR MENU =====
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home – Giới thiệu đề tài",
        "Analysis – Xử lý dữ liệu & Train model",
        "Training Info – Thông tin huấn luyện"
    ]
)

# ===== ROUTING =====

if page.startswith("Home"):
    from modules.Home import show
    show()

elif page.startswith("Analysis"):
    from modules.Analysis import show
    show()

elif page.startswith("Training Info"):
    from modules.Training_Info import show
    show()


# ===== FOOTER =====
# ===== FOOTER =====
st.write("---")

# --- STUDENTS BOX ---
st.markdown(
    """
    <div style="padding:18px; background:#ffffdd; border-radius:10px;">
        <b>Students:</b><br>
        - Student 1: ... email<br>
        - Student 2: ... email<br>
        - Student 3: ... email<br>
        - Student 4: ... email<br>
    </div>
    """,
    unsafe_allow_html=True
)

# --- INSTRUCTOR BOX ---
st.markdown(
    """
    <div style="
        padding:18px;
        background:#fafafa;
        border-radius:12px;
        border:1px solid #ddd;
        margin-top:15px;
        font-size:16px;
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"
             width="22"
             style="vertical-align:middle; margin-right:6px;">
        <b>Bùi Tiến Đức</b> _
        <a href="https://orcid.org/0000-0001-5174-3558"
           target="_blank"
           style="text-decoration:none; color:#0073e6;">
           ORCID Profile _ https://orcid.org/0000-0001-5174-3558
        </a> 
    </div>
    """,
    unsafe_allow_html=True
)

