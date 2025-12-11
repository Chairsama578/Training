import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import io
import re
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

import joblib


# =========================================================
# 🔧 TIỀN XỬ LÝ TIẾNG VIỆT
# =========================================================

STOPWORDS = set("""
và là của những cái các một trong được để với từ khi mà thì là đều này kia hoặc nên nếu tuy vì nhưng vậy còn rất lại đã đang sẽ
""".split())

def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)

def clean_regex(text):
    text = re.sub(r"[^a-zA-Z0-9À-ỹ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_stopwords(words):
    return " ".join([w for w in words.split() if w not in STOPWORDS])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.lower()
    text = clean_regex(text)
    text = remove_stopwords(text)
    return text


# =========================================================
# 📦 HÀM ĐỌC DATA TỪ ZIP MẪU
# =========================================================

def generate_sample_zip():
    files = {
        "politics_01.txt": "Chính phủ vừa thông qua nghị quyết mới về phát triển kinh tế số.",
        "education_01.txt": "Bộ Giáo dục công bố đổi mới chương trình học phổ thông.",
        "weather_01.txt": "Miền Bắc rét đậm do ảnh hưởng không khí lạnh.",
        "sports_01.txt": "Việt Nam thắng 3-1 trong trận giao hữu."
    }

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as z:
        for fn, content in files.items():
            z.writestr(fn, content)

    mem.seek(0)
    return mem


# =========================================================
# 📥 ĐỌC FOLDER TXT – BẢN PRO
# =========================================================

def read_txt_folder(files):
    rows = []

    for f in files:
        if f.name.endswith(".txt"):

            # Tách nhãn từ tên file
            # Ví dụ "1. SPORTS.txt" -> "SPORTS"
            base = os.path.splitext(f.name)[0]
            parts = base.split(".")
            label = parts[-1].strip().upper()

            # Đọc file và tách mỗi dòng thành 1 mẫu
            content = f.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            for line in lines:
                rows.append([line, label])

    return pd.DataFrame(rows, columns=["text", "label"])


# =========================================================
# 📥 ĐỌC ZIP – TỰ NHẬN LABEL
# =========================================================

def read_txt_zip(file):
    rows = []
    with zipfile.ZipFile(file, "r") as z:

        for fn in z.namelist():
            if fn.endswith(".txt"):

                base = os.path.splitext(fn)[0]
                parts = base.split("_")
                label = parts[0].upper()

                text = z.read(fn).decode("utf-8", errors="ignore")
                lines = [line.strip() for line in text.split("\n") if line.strip()]

                for line in lines:
                    rows.append([line, label])

    return pd.DataFrame(rows, columns=["text", "label"])


# =========================================================
# 🧠 GIAO DIỆN CHÍNH
# =========================================================

def show():

    st.markdown("### 🧠 Analysis – Train mô hình phân loại tin tức (Bản PRO)")

    st.download_button(
        "⬇️ Tải ZIP mẫu (4 mẫu nhỏ)",
        data=generate_sample_zip(),
        file_name="sample_news.zip",
        mime="application/zip"
    )

    st.write("---")
    st.header("1️⃣ Upload dữ liệu")

    mode = st.radio(
        "Chọn chế độ tải dữ liệu:",
        ["Folder TXT", "ZIP TXT", "CSV / Excel"],
        horizontal=True
    )

    if "df" not in st.session_state:
        st.session_state.df = None

    df = None

    # --- FOLDER TXT ---
    if mode == "Folder TXT":
        files = st.file_uploader("Chọn nhiều file TXT", type=["txt"], accept_multiple_files=True)
        if files:
            df = read_txt_folder(files)
            st.session_state.df = df
            st.success(f"✔ Đã đọc {len(df)} dòng tin tức!")
            st.dataframe(df)

    # --- ZIP TXT ---
    elif mode == "ZIP TXT":
        up = st.file_uploader("Upload ZIP", type=["zip"])
        if up:
            df = read_txt_zip(up)
            st.session_state.df = df
            st.success(f"✔ ZIP đã đọc thành công ({len(df)} dòng)!")
            st.dataframe(df)

    # --- CSV / EXCEL ---
    else:
        up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if up:
            ext = up.name.split(".")[-1]
            df = pd.read_csv(up) if ext == "csv" else pd.read_excel(up)
            st.session_state.df = df
            st.success("✔ File bảng đã đọc thành công!")
            st.dataframe(df)

    st.write("---")

    # =========================================================
    # 📊 PHÂN TÍCH NHANH DATASET
    # =========================================================
    if st.session_state.df is not None:

        st.subheader("📊 Thống kê dữ liệu theo Label")

        # fig, ax = plt.subplots(figsize=(6, 4))
        # sns.countplot(x=st.session_state.df["label"], ax=ax)
        # plt.xticks(rotation=45)
        # st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(4, 2.5))   # Thu nhỏ figure
        sns.countplot(x=st.session_state.df["label"], ax=ax)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Quan trọng: không cho Streamlit phóng rộng!
        st.pyplot(fig, use_container_width=False)


    st.write("---")
    st.header("2️⃣ Train model")

    model_choice = st.selectbox(
        "Chọn model:",
        ["Auto (XGBoost)", "XGBoost", "Logistic Regression", "SVM"]
    )

    status = st.empty()

    # =========================================================
    # 🚀 TRAIN MODEL
    # =========================================================

    if st.button("🚀 Train"):

        df = st.session_state.df
        if df is None or len(df) < 10:
            st.error("❌ Dataset quá nhỏ. Cần ít nhất 10 dòng tin tức.")
            return

        # Tiền xử lý text
        df["text"] = df["text"].apply(preprocess_text)

        # Label Encoder
        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        X = df["text"].values

        # Kiểm tra số mẫu mỗi class
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            st.error("❌ Mỗi nhãn cần tối thiểu 2 mẫu để train.")
            return

        # TF-IDF
        status.info("🔄 Đang tạo đặc trưng TF-IDF...")

        vectorizer = TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1
        )
        X_vec = vectorizer.fit_transform(X)

        # Chọn test size tự động
        test_size = 0.25
        stratify_flag = y if class_counts.min() >= 3 else None

        status.info("🔥 Training... vui lòng đợi...")

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y,
            test_size=test_size,
            stratify=stratify_flag,
            random_state=42
        )

        # MODEL
        if model_choice in ["Auto (XGBoost)", "XGBoost"]:
            model = XGBClassifier(
                n_estimators=350,
                learning_rate=0.08,
                max_depth=10,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss"
            )
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=5000)
        else:
            model = SVC(kernel="linear", probability=True)

        # TRAIN
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        status.success(f"🎯 Accuracy: **{acc:.4f}**")

        # SAVE MODEL
        os.makedirs("export", exist_ok=True)
        joblib.dump(model, "export/model.pkl")
        joblib.dump(vectorizer, "export/vectorizer.pkl")
        joblib.dump(le, "export/label_encoder.pkl")

        st.success("📦 Model đã lưu thành công vào thư mục export/!")

    # =========================================================
    # 🔮 DỰ BÁO
    # =========================================================
    st.write("---")
    st.header("3️⃣ Dự báo")

    txt = st.text_area("Nhập nội dung tin tức...")

    if st.button("🔮 Dự báo"):
        if not os.path.exists("export/model.pkl"):
            st.error("❌ Chưa có model. Hãy train trước.")
            return

        model = joblib.load("export/model.pkl")
        vec = joblib.load("export/vectorizer.pkl")
        le = joblib.load("export/label_encoder.pkl")

        vec_txt = vec.transform([preprocess_text(txt)])
        pred = model.predict(vec_txt)[0]
        label = le.inverse_transform([pred])[0]

        st.success(f"➡ Kết quả dự báo: **{label}**")
