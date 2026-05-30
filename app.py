import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# 1. THIẾT LẬP CẤU HÌNH TRANG WEB
st.set_page_config(page_title="ESSRC Beam Predictor", layout="centered")
st.title("Flexural Capacity Prediction of ECC Layer and Steel Plate Strengthened RC (ESSRC) Beams")
st.write(" ")

# 2. TẢI MÔ HÌNH VÀ DỮ LIỆU CHUẨN HÓA (MIN/MAX)
@st.cache_resource
def load_ml_components():
    # Load mô hình và scaler theo đúng tên file trong thư mục của bạn
    # THÊM compile=False VÀO ĐÂY:
    model = tf.keras.models.load_model('essrc_ann_model.h5', compile=False)
    scaler_data = joblib.load('scaler_manual.pkl')
    return model, scaler_data

try:
    model, scaler_data = load_ml_components()
    X_min = scaler_data['X_min']
    X_max = scaler_data['X_max']
    y_min = scaler_data['y_min']
    y_max = scaler_data['y_max']
except Exception as e:
    st.error(f"Lỗi khi tải file cấu hình: {e}")
    st.stop()

# 3. GIAO DIỆN NHẬP THÔNG SỐ (Chia làm 2 cột cho cân đối)
st.subheader("Khai báo thông số đầu vào")
col1, col2 = st.columns(2)

with col1:
    b = st.number_input("Bề rộng b (mm)", value=150.0, step=10.0)
    hc = st.number_input("Chiều cao hc (mm)", value=170.0, step=10.0)
    hECC = st.number_input("Chiều cao hECC (mm)", value=30.0, step=5.0)
    us = st.number_input("Tỷ lệ cốt thép μs (%)", value=1.577, step=0.1)
    fc_c = st.number_input("Cường độ bê tông f'c,c (MPa)", value=17.12, step=1.0)

with col2:
    Ec_c = st.number_input("Mô đun đàn hồi Ec,c (MPa)", value=20000.0, step=1000.0)
    fc_ECC = st.number_input("Cường độ f'c,ECC (MPa)", value=30.76, step=1.0)
    Ec_ECC = st.number_input("Mô đun đàn hồi Ec,ECC (MPa)", value=15500.0, step=500.0)
    tb_p = st.number_input("Bề dày tấm đỉnh/đáy tb,p (mm)", value=50.0, step=5.0)
    tw_p = st.number_input("Bề dày vách thép tw,p (mm)", value=4.0, step=0.5)

# 4. XỬ LÝ TÍNH TOÁN KHI BẤM NÚT DỰ ĐOÁN
st.write("---")
if st.button("🚀 Tính toán kết quả Mu", type="primary"):
    # Tạo mảng chứa 10 thông số đầu vào theo đúng thứ tự tập train ban đầu
    X_new = np.array([[b, hc, hECC, us, fc_c, Ec_c, fc_ECC, Ec_ECC, tb_p, tw_p]])
    
    # Thực hiện chuẩn hóa Min-Max thủ công
    X_scaled = (X_new - X_min) / (X_max - X_min)
    
    # Dự đoán giá trị đã chuẩn hóa qua mạng ANN
    y_pred_scaled = model.predict(X_scaled)
    
    # Khôi phục giá trị thực (Inverse Transform) cho biến đầu ra Mu
    Mu_real = y_pred_scaled[0][0] * (y_max - y_min) + y_min
    
    # Hiển thị kết quả ra màn hình với định dạng đẹp mắt
    st.success("Tính toán hoàn tất!")
    st.metric(label="Sức kháng uốn dự đoán (Mu)", value=f"{Mu_real:.4f} kNm")
