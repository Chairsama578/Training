import streamlit as st

def show():

    st.markdown("<h3>Giới thiệu Đề tài</h3>", unsafe_allow_html=True)

    st.write(
        """
        **Topic 2: Developing a Vietnamese News Classification System Using XGBoost**

        Mục tiêu của đề tài:
        - Thu thập và xử lý dữ liệu tin tức tiếng Việt từ nhiều nguồn.
        - Chuyển đổi văn bản thành vector TF-IDF.
        - Huấn luyện mô hình phân loại tin tức bằng XGBoost.
        - Cho phép người dùng tải lên CSV, Excel hoặc file .txt.
        - Hệ thống tự động đề xuất mô hình tốt nhất và cho phép người dùng chọn mô hình khác.
        - Lưu model.pkl sau khi huấn luyện.
        - Cho phép người dùng nhập tin tức mới và dự báo nhãn.
        """
    )

    st.image("rose.png", width=300)
