import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# ==========================================
# 1. TÙY CHỈNH CSS (Giữ nguyên từ app cũ của bạn)
# ==========================================
st.markdown(
    """
    <style>
    /* 1. Kích thước chữ của các mục lớn (st.subheader) */
    h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #4da6ff !important; 
    }

    /* 2. Kích thước chữ của các mục nhỏ (Label của Slider, Number Input và Selectbox) */
    .stSlider label p, 
    .stSelectbox label p,
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] p {
        font-size: 24px !important; 
    }
    
    /* 3. Tùy chỉnh khoảng cách thừa cho gọn */
    .stSlider, .stSelectbox, .stNumberInput {
        margin-top: -10px !important; 
    }

    /* 4. Cỡ chữ của các CON SỐ trên widget */
    [data-baseweb="slider"] div, 
    [data-baseweb="slider"] span,
    input[type="number"] {
        font-size: 22px !important; 
        font-weight: 600 !important; 
    }
    
    /* 5. CỠ CHỮ BÊN TRONG HỘP CHỌN (SELECTBOX) */
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li span {
        font-size: 20px !important; 
        font-weight: 600 !important; 
    }

    /* 6. Cỡ chữ của các hộp TEXT tự động tính toán (st.info) */
    div[data-testid="stAlert"] p {
        font-size: 24px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 2. CẤU HÌNH TRANG VÀ TIÊU ĐỀ
# ==========================================
st.set_page_config(page_title="ESSRC Beam Predictor", layout="wide")
st.title("Prediction of Flexural Capacity of ESSRC Beams")
st.markdown("##### **Developed by:** [Tên của bạn/Nhóm nghiên cứu của bạn]")
st.markdown("---")

# ==========================================
# 3. TẢI MÔ HÌNH VÀ SCALER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'essrc_ann_model.h5')
    scaler_path = os.path.join(BASE_DIR, 'scaler_manual.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Cannot find: {scaler_path}")
        
    model = tf.keras.models.load_model(model_path, compile=False) 
    scaler_data = joblib.load(scaler_path)
    
    X_min = scaler_data['X_min']
    X_max = scaler_data['X_max']
    y_min = scaler_data['y_min']
    y_max = scaler_data['y_max']
    
    return model, X_min, X_max, y_min, y_max

try:
    model, X_min, X_max, y_min, y_max = load_assets()
except Exception as e:
    st.error(f"INITIALIZATION ERROR: {e}")
    st.stop()

# ==========================================
# 4. GIAO DIỆN NHẬP LIỆU (EXPANDER & 3 CỘT)
# ==========================================
with st.expander("**⚙️ INPUT PARAMETERS (Click to Expand / Collapse)**", expanded=True):
    
    col1, col2, col3 = st.columns(3)
    
    # --- CỘT 1: Thông số Hình học ---
    with col1:
        st.subheader("1. Geometry Configuration")
        b = st.number_input("Beam width - b (mm)", value=150.0, step=10.0)
        hc = st.number_input("Concrete height - hc (mm)", value=170.0, step=10.0)
        hECC = st.number_input("ECC height - hECC (mm)", value=30.0, step=5.0)
        
        # Ví dụ tự động tính toán tổng chiều cao
        h_total = hc + hECC
        st.info(f"💡 Total section height (h) auto-calculated: **{h_total:.1f} mm**")

    # --- CỘT 2: Thông số Vật liệu Bê tông & ECC ---
    with col2:
        st.subheader("2. Material Properties")
        fc_c = st.number_input("NC compressive strength - f'c,c (MPa)", value=17.12, step=1.0)
        Ec_c = st.number_input("NC elastic modulus - Ec,c (MPa)", value=20000.0, step=1000.0)
        fc_ECC = st.number_input("ECC compressive strength - f'c,ECC (MPa)", value=30.76, step=1.0)
        Ec_ECC = st.number_input("ECC elastic modulus - Ec,ECC (MPa)", value=15500.0, step=500.0)

    # --- CỘT 3: Cốt thép & Vách thép ---
    with col3:
        st.subheader("3. Steel & Reinforcement")
        us = st.number_input("Reinforcement ratio - μs (%)", value=1.577, step=0.1, format="%.3f")
        tb_p = st.number_input("Top/Bottom plate thickness - tb,p (mm)", value=50.0, step=5.0)
        tw_p = st.number_input("Web plate thickness - tw,p (mm)", value=4.0, step=0.5)

    st.markdown("---")
    
    # Nút bấm chạy dự đoán
    run_button = st.button("🚀 RUN PREDICTION", use_container_width=True)

# ==========================================
# 5. XỬ LÝ DỮ LIỆU & IN KẾT QUẢ
# ==========================================
# Gom 10 thông số theo ĐÚNG thứ tự lúc train
input_data = np.array([[b, hc, hECC, us, fc_c, Ec_c, fc_ECC, Ec_ECC, tb_p, tw_p]])

if run_button:
    with st.spinner("Analyzing data..."):
        # Chuẩn hóa Min-Max
        input_scaled = (input_data - X_min) / (X_max - X_min)
        
        # Dự đoán
        prediction_norm = model.predict(input_scaled)
        
        # Giải chuẩn hóa
        prediction_real = prediction_norm[0][0] * (y_max - y_min) + y_min
        
        st.success("Prediction Completed!")
        
        # Khung hiển thị kết quả
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            # LƯU Ý: Bạn hãy thay số 2.15 này bằng giá trị MAE thực tế của mô hình bạn
            MAE_error = 2.15 
            
            st.markdown("<p style='font-size: 24px; font-weight: bold; margin-bottom: 0px;'>Predicted Flexural Capacity of ESSRC Beam (Mu)</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='display: flex; align-items: baseline; gap: 8px; margin-bottom: 0px;'>"
                f"<span style='font-size: 40px; font-weight: bold;'>{prediction_real:.2f} kNm</span>"
                f"<span style='font-size: 16px; color: #A5A5A5;'>± {MAE_error} kNm Expected Error (MAE)</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            # LƯU Ý: Đổi hệ số R2 cho đúng với bài của bạn
            st.markdown("<p style='font-size: 14px; color: #09AB3B; margin-top: 5px;'>↑ ANN Model (R² = 0.98)</p>", unsafe_allow_html=True)
        
        with col_res2:
             st.info(f"Target Design Status: Ready for review")

# ==========================================
# 6. HIỂN THỊ HÌNH ẢNH MÔ TẢ & SHAP (NẾU CÓ)
# ==========================================
st.markdown("---")
st.markdown("### Explainable AI")

# Thay đổi tên file ảnh 'essrc_shap_info.png' cho đúng với ảnh bạn đưa vào thư mục nhé
image_path = os.path.join(BASE_DIR, 'essrc_shap_info.PNG')

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True, caption="Geometric parameters and SHAP-based feature importance analysis")
else:
    st.warning("Vui lòng thêm file ảnh 'essrc_shap_info.png' vào thư mục để hiển thị hình minh họa.")
