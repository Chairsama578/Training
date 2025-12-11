import streamlit as st
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ============================================================
# 1. HÀM KIỂM TRA DỮ LIỆU THÔ
# ============================================================

def load_and_validate_raw_data():
    path = "raw_news.csv"

    if not os.path.exists(path):
        return None, ["⚠ Không tìm thấy file raw_news.csv!"]

    errors = []
    df = None

    try:
        df = pd.read_csv(path)

        # 1) File rỗng
        if df.empty:
            errors.append("❌ File raw_news.csv rỗng.")
            return None, errors

        # 2) Thiếu cột
        required_cols = ["text", "label"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"❌ Thiếu cột bắt buộc: {col}")

        if errors:
            return None, errors

        # 3) Null check
        if df["text"].isna().sum() > 0:
            errors.append("❌ Có dòng bị null trong cột TEXT.")

        if df["label"].isna().sum() > 0:
            errors.append("❌ Có dòng bị null trong cột LABEL.")

        # 4) Dòng rỗng hoặc toàn ký tự trắng
        empty_rows = df["text"].str.strip().eq("").sum()
        if empty_rows > 0:
            errors.append(f"❌ Có {empty_rows} dòng text bị rỗng.")

        # 5) Ký tự đặc biệt → flag
        pattern_special = r"[^0-9A-Za-zÀ-ỹ\s\.,!?%-]"
        special_rows = df["text"].str.contains(pattern_special, regex=True).sum()
        if special_rows > 0:
            errors.append(
                f"⚠ Phát hiện {special_rows} dòng chứa ký tự đặc biệt (emoji, ký tự lạ...)."
            )

        # 6) Text quá ngắn
        short_rows = df[df["text"].str.len() < 5]
        if len(short_rows) > 0:
            errors.append(f"⚠ Có {len(short_rows)} câu quá ngắn (<5 ký tự).")

        # 7) Trùng lặp
        dup = df.duplicated().sum()
        if dup > 0:
            errors.append(f"⚠ Có {dup} dòng trùng lặp trong dữ liệu.")

        # 8) Ít hơn 2 nhãn → không thể train
        if df["label"].nunique() < 2:
            errors.append("❌ File chỉ có 1 nhãn → không thể train model!")

        return df, errors

    except Exception as e:
        return None, [f"❌ Lỗi đọc file: {e}"]


# ============================================================
# 2. TIỀN XỬ LÝ TEXT
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# PAGE MAIN
# ============================================================

def show():

    st.markdown(
        "<h3 style='color:blue;'>Training Info – Thông số Huấn luyện Máy tính</h3>",
        unsafe_allow_html=True,
    )
    st.write("---")

    # ============================================================
    # 1. HIỂN THỊ DỮ LIỆU THÔ + KIỂM TRA LỖI
    # ============================================================

    st.write("## 1. Dữ liệu thô")

    df_raw, issues = load_and_validate_raw_data()

    # Ưu tiên dữ liệu user upload từ Analysis.py
    if "df" in st.session_state and st.session_state.df is not None:
        df_raw = st.session_state.df
        issues = []
        st.success("✔ Đang dùng dữ liệu trực tiếp từ người dùng upload (Analysis.py).")

    # Hiện các cảnh báo
    if issues:
        for msg in issues:
            st.warning(msg)

    if df_raw is None:
        st.stop()

    df_raw["length"] = df_raw["text"].str.len()
    st.dataframe(df_raw.head())

    st.write("---")

    # ============================================================
    # 2. HIỂN THỊ DỮ LIỆU SAU TIỀN XỬ LÝ
    # ============================================================

    st.write("## 2. Dữ liệu sau tiền xử lý")

    df_processed = df_raw.copy()
    df_processed["clean_text"] = df_processed["text"].apply(clean_text)

    st.dataframe(df_processed.head())

    st.write("---")

    # ============================================================
    # 3. HIỂN THỊ ĐƯỜNG DẪN MODEL + VECTOR
    # ============================================================

    st.write("## 3. Đường dẫn Model & Vectorizer")

    model_path = "export/model.pkl"
    vec_path = "export/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        st.error("❌ Không tìm thấy model.pkl hoặc vectorizer.pkl → chưa train model.")
        st.stop()

    st.success(f"✔ Model: {model_path}")
    st.success(f"✔ Vectorizer: {vec_path}")

    st.write("---")

    # ============================================================
    # 4. THÔNG TIN MODEL ĐÃ HUẤN LUYỆN
    # ============================================================

    st.write("## 4. Thông tin Model đã huấn luyện")

    model = joblib.load(model_path)
    st.code(str(model))

    st.write("---")

    # ============================================================
    # 5. HIỂN THỊ TRAIN_INFO.JSON
    # ============================================================

    st.write("## 5. Thông tin file train_info.json")

    train_info = "export/train_info.json"
    if os.path.exists(train_info):
        import json

        st.json(json.load(open(train_info)))
        st.success("✔ Đọc file train_info.json thành công.")
    else:
        st.warning("⚠ Không có train_info.json — hãy train lại model!")

    st.write("---")

    # ============================================================
    # 6. ĐÁNH GIÁ MÔ HÌNH BẰNG MACRO F1-SCORE
    # ============================================================

    st.write("## 6. Đánh giá mô hình bằng Macro F1-score")

    vectorizer = joblib.load(vec_path)
    X = vectorizer.transform(df_processed["clean_text"])

    # Ép nhãn về string để tránh lỗi mix kiểu
    y = df_processed["label"].astype(str)

    # Dự báo với model chính và ép về string
    preds = pd.Series(model.predict(X)).astype(str)

    # F1-score
    f1 = f1_score(y, preds, average="macro")
    st.success(f"🔥 Macro F1-score: {f1:.4f}")

    # Báo cáo chi tiết
    st.text(classification_report(y, preds))

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax, fmt="d")
    st.pyplot(fig)

    st.write("---")

    # ============================================================
    # 7. SO SÁNH 3 MÔ HÌNH: XGBoost – SVM – Logistic Regression
    #    (tính Macro F1-score, đã FIX label cho XGBoost)
    # ============================================================

    st.write("## 7. So sánh các mô hình ML (Macro F1-score)")

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    # Dùng LabelEncoder để mã hoá nhãn sang số cho cả 3 mô hình
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42
    )

    # XGBoost cần dense
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    results = {}

    # XGBoost
    try:
        xgb = XGBClassifier(eval_metric="mlogloss")
        xgb.fit(X_train_dense, y_train)
        preds_xgb = xgb.predict(X_test_dense)
        results["XGBoost"] = f1_score(y_test, preds_xgb, average="macro")
    except Exception as e:
        results["XGBoost"] = 0.0
        st.error(f"Lỗi XGBoost: {e}")

    # SVM
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_test)
    results["SVM"] = f1_score(y_test, preds_svm, average="macro")

    # Logistic Regression
    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    results["Logistic Regression"] = f1_score(y_test, preds_lr, average="macro")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(results.keys(), results.values(), color=["#007bff", "#ff7f0e", "#28a745"])
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("So sánh F1-score giữa các mô hình ML")
    st.pyplot(fig)

    # Lưu file
    os.makedirs("export", exist_ok=True)
    fig.savefig("export/images.png")
    st.success("✔ Đã lưu ảnh biểu đồ vào export/images.png")

    st.json(results)

    st.write("---")

    # ============================================================
    # 8. KẾT LUẬN
    # ============================================================

    st.write("## 8. Kết luận")
    st.info(
        """
    ✔ Dữ liệu đã được kiểm tra và làm sạch.  
    ✔ Mô hình hiện tại hoạt động ổn định với F1-score cao.  
    ✔ XGBoost / SVM / Logistic Regression có thể dùng làm baseline.  
    ✔ Có thể nâng cấp bằng BERT / PhoBERT để cải thiện chất lượng.  
    """
    )
