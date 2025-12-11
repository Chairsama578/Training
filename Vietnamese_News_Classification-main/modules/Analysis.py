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

            base = os.path.splitext(f.name)[0]
            parts = base.split(".")
            label = parts[-1].strip().upper()

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

    # === CHỨC NĂNG MỚI: DOWNLOAD CSV ===
    if st.session_state.df is not None:
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Tải xuống dữ liệu CSV",
            data=csv,
            file_name="dataset.csv",
            mime="text/csv"
        )

    st.write("---")


    # =========================================================
    # 📊 PHÂN TÍCH NHANH DATASET
    # =========================================================
    if st.session_state.df is not None:

        st.subheader("📊 Thống kê dữ liệu theo Label")

        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.countplot(x=st.session_state.df["label"], ax=ax)

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)

        # =========================================================
        # ⭐ EXTRA FUNCTION 1 — Histogram độ dài câu
        # =========================================================
        st.subheader("📏 Phân phối độ dài câu (Histogram)")

        st.session_state.df["length"] = st.session_state.df["text"].apply(lambda x: len(str(x).split()))

        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        ax2.hist(st.session_state.df["length"], bins=20, color="#1abc9c", edgecolor="black")
        ax2.set_title("Phân phối số lượng từ trong câu")
        ax2.set_xlabel("Số từ")
        ax2.set_ylabel("Số câu")
        plt.tight_layout()

        st.pyplot(fig2, use_container_width=False)

        # =========================================================
        # ⭐ EXTRA FUNCTION 2 — WordCloud cho từng label
        # =========================================================
        from wordcloud import WordCloud

        st.subheader("☁️ WordCloud theo từng nhãn")

        labels = st.session_state.df["label"].unique()

        for lb in labels:
            subset = st.session_state.df[st.session_state.df["label"] == lb]

            text_blob = " ".join(subset["text"].astype(str).tolist())

            if len(text_blob.strip()) < 5:
                st.warning(f"⚠ Không đủ dữ liệu để tạo WordCloud cho label: {lb}")
                continue

            wc = WordCloud(
                width=600,
                height=300,
                background_color="white",
                colormap="viridis"
            ).generate(text_blob)

            fig_wc, ax_wc = plt.subplots(figsize=(6, 3))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")

            st.markdown(f"### 🏷 WordCloud – {lb}")
            st.pyplot(fig_wc, use_container_width=False)

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

        df["text"] = df["text"].apply(preprocess_text)

        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        X = df["text"].values

        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            st.error("❌ Mỗi nhãn cần tối thiểu 2 mẫu để train.")
            return

        status.info("🔄 Đang tạo đặc trưng TF-IDF...")

        vectorizer = TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            min_df=1
        )
        X_vec = vectorizer.fit_transform(X)

        stratify_flag = y if class_counts.min() >= 3 else None

        status.info("🔥 Training... vui lòng đợi...")

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y,
            test_size=0.25,
            stratify=stratify_flag,
            random_state=42
        )

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

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        status.success(f"🎯 Accuracy: **{acc:.4f}**")

        os.makedirs("export", exist_ok=True)
        joblib.dump(model, "export/model.pkl")
        joblib.dump(vectorizer, "export/vectorizer.pkl")
        joblib.dump(le, "export/label_encoder.pkl")

        st.success("📦 Model đã lưu thành công vào thư mục export/!")

        # =========================================================
        # ⭐ EXTRA FUNCTION 3 — EXPORT MODEL ZIP
        # =========================================================
        import zipfile

        with zipfile.ZipFile("export/model_package.zip", "w") as z:
            z.write("export/model.pkl")
            z.write("export/vectorizer.pkl")
            z.write("export/label_encoder.pkl")

        with open("export/model_package.zip", "rb") as f:
            st.download_button(
                "📥 Tải Model.zip",
                data=f,
                file_name="model_package.zip",
                mime="application/zip"
            )

        # =========================================================
        # ⭐ EXTRA FUNCTION 4 — Training Report
        # =========================================================
        from sklearn.metrics import confusion_matrix, classification_report
        import json

        # Accuracy bar chart
        fig_acc, ax_acc = plt.subplots(figsize=(4, 2.5))
        ax_acc.bar(["Accuracy"], [acc], color="#1abc9c")
        ax_acc.set_ylim(0, 1)
        plt.tight_layout()
        fig_acc.savefig("export/accuracy.png")
        st.image("export/accuracy.png", caption="🎯 Accuracy", width=450)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_
        )
        ax_cm.set_title("Confusion Matrix")
        plt.tight_layout()
        fig_cm.savefig("export/confusion_matrix.png")
        st.image("export/confusion_matrix.png", caption="📌 Confusion Matrix", width=450)

        # Text report
        report = classification_report(y_test, preds, target_names=le.classes_)
        with open("export/report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        st.code(report, language="text")

        # Save training metadata
        train_info = {
              "accuracy": float(acc),
              "model_name": str(model.__class__.__name__),
              "num_samples": len(df),
              "train_size": X_train.shape[0],   # FIXED
              "test_size": X_test.shape[0],     # FIXED
              "labels": list(le.classes_)
        }


        with open("export/train_info.json", "w", encoding="utf-8") as f:
            json.dump(train_info, f, indent=4, ensure_ascii=False)

        st.success("📁 Đã lưu đầy đủ file báo cáo tại export/")


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

    # =========================================================
    # ⭐ EXTRA FUNCTION 5 — BATCH PREDICTION
    # =========================================================
    st.subheader("📂 Dự báo hàng loạt từ file CSV / TXT")

    batch_file = st.file_uploader("Upload file dự báo", type=["txt", "csv"])

    if batch_file:
        ext = batch_file.name.split(".")[-1].lower()

        if ext == "txt":
            lines = batch_file.read().decode("utf-8").split("\n")
            df_batch = pd.DataFrame({"text": [l.strip() for l in lines if l.strip()]})

        elif ext == "csv":
            df_batch = pd.read_csv(batch_file)

        else:
            st.error("❌ Chỉ hỗ trợ TXT hoặc CSV.")
            return

        model = joblib.load("export/model.pkl")
        vec = joblib.load("export/vectorizer.pkl")
        le = joblib.load("export/label_encoder.pkl")

        df_batch["clean"] = df_batch["text"].apply(preprocess_text)
        Xb = vec.transform(df_batch["clean"])
        preds = model.predict(Xb)
        df_batch["label"] = le.inverse_transform(preds)

        st.dataframe(df_batch)

        csv_out = df_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Tải kết quả dự báo",
            data=csv_out,
            file_name="batch_prediction.csv",
            mime="text/csv"
        )
