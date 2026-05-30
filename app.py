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
st.title("Flexural Capacity Prediction of ECC Layer and Steel Plate Strengthened RC (ESSRC) Beams")
st.markdown("##### **Developed by:** ")
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
# 4. GIAO DIỆN NHẬP LIỆU (EXPANDER & 3 CỘT ĐỒNG BỘ SLIDER)
# ==========================================
with st.expander("**⚙️ INPUT PARAMETERS (Click to Expand / Collapse)**", expanded=True):
    
    col1, col2, col3 = st.columns(3)
    
    # --- CỘT 1: Thông số Hình học (DÙNG SLIDER) ---
    with col1:
        st.subheader("1. Geometry Configuration")
        b = st.slider("Beam width - b (mm)", min_value=100, max_value=250, value=150, step=10)
        hc = st.slider("Concrete height - hc (mm)", min_value=100, max_value=250, value=170, step=10)
        hECC = st.slider("ECC height - hECC (mm)", min_value=20, max_value=150, value=30, step=5)
        
        h_total = hc + hECC
        st.info(f"💡 Total section height (h) auto-calculated: **{h_total} mm**")

    # --- CỘT 2: Thông số Vật liệu Bê tông & ECC (SLIDER TRONG KHOẢNG TRAIN) ---
    with col2:
        st.subheader("2. Material Properties")
        
        # 1. Khai báo f'c,c
        fc_c = st.slider("NC compressive strength - f'c,c (MPa)", 
                         min_value=17.12, max_value=27.54, value=17.12, 
                         step=0.01, format="%.2f")
        
        Ec_c = int(np.interp(fc_c, [17.12, 27.54], [20000, 25000]))
        st.info(f"💡 NC elastic modulus - Ec,c auto-calculated: **{Ec_c} MPa**")
        
        st.markdown("---")
        
        # 2. Khai báo f'c,ECC
        fc_ECC = st.slider("ECC compressive strength - f'c,ECC (MPa)", 
                           min_value=30.76, max_value=52.30, value=30.76, 
                           step=0.01, format="%.2f")
        
        Ec_ECC = int(np.interp(fc_ECC, 
                               [30.76, 43.07, 52.30], 
                               [15500, 17000, 18000]))
        st.info(f"💡 ECC elastic modulus - Ec,ECC auto-calculated: **{Ec_ECC} MPa**")

    # --- CỘT 3: Cốt thép & Vách thép (DÙNG SLIDER) ---
    with col3:
        st.subheader("3. Steel & Reinforcement")
        
        us = st.slider("Reinforcement ratio - μs (%)", 
                       min_value=0.500, max_value=3.000, value=1.577, 
                       step=0.001, format="%.3f")
                       
        tb_p = st.slider("Top/Bottom plate thickness - tb,p (mm)", 
                         min_value=20, max_value=100, value=50, step=5)
                         
        tw_p = st.slider("Web plate thickness - tw,p (mm)", 
                         min_value=2.0, max_value=10.0, value=4.0, 
                         step=0.5, format="%.1f")

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
             st.info(f"Overall Dimensions (b x h x L): **{b} x {h_total} x 2000 mm**")

# ==========================================
# 6. HIỂN THỊ HÌNH ẢNH MÔ TẢ & SHAP (NẾU CÓ)
# ==========================================
st.markdown("---")
st.markdown("### Explainable AI")

# Thay đổi tên file ảnh 'essrc_shap_info.png' cho đúng với ảnh bạn đưa vào thư mục nhé
image_path = os.path.join(BASE_DIR, 'essrc_shap_info.PNG')

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True, caption="Explainable AI SHAP-based feature importance analysis")
else:
    st.warning("Vui lòng thêm file ảnh 'essrc_shap_info.png' vào thư mục để hiển thị hình minh họa.")
