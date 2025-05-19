import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pickle

@st.cache_resource
def load_model():
    model = pickle.load(open("xgb_model.pkl", "rb"))
    encoder = pickle.load(open("ordinal_encoder.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    feature_order = pickle.load(open("feature_order.pkl", "rb"))
    return model, encoder, label_encoder, feature_order

st.title("Heart Disease Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dữ liệu upload:")
    st.dataframe(df.head())

    model, encoder, label_encoder, feature_order = load_model()

    X = df.copy()
    if 'HeartDisease' in X.columns:
        X = X.drop(columns=['HeartDisease'])

    # Đảm bảo đúng thứ tự và tên cột như khi train
    try:
        X = X[feature_order]
    except Exception as e:
        st.error(f"Thiếu hoặc sai tên cột so với dữ liệu train: {e}")
        st.stop()

    # Danh sách các cột phân loại và số
    categorical_cols = [
        'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
        'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'
    ]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Ép kiểu về string cho các cột phân loại
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)

    # Kiểm tra giá trị mới với các cột phân loại
    for i, col in enumerate(categorical_cols):
        if col in X.columns:
            train_cats = set([str(x) for x in encoder.categories_[i]])
            upload_cats = set(X[col].unique())
            unseen = upload_cats - train_cats
            if unseen:
                st.error(f"Cột '{col}' có giá trị mới chưa từng xuất hiện khi train: {unseen}")
                st.stop()

    # Encode categorical, giữ nguyên numerical
    X_cat = encoder.transform(X[categorical_cols])
    X_num = X[numerical_cols].values
    X_all = np.concatenate([X_num, X_cat], axis=1)

    # Dự đoán
    y_pred = model.predict(X_all)
    y_pred_label = label_encoder.inverse_transform(y_pred)

    df['HeartDisease_Prediction'] = y_pred_label
    st.write("Kết quả dự đoán:")
    st.dataframe(df)
    st.download_button("Tải kết quả về", df.to_csv(index=False), file_name="prediction_result.csv")