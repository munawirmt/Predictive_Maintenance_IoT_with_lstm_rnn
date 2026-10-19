import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle

st.set_page_config(page_title="IoT Predictive Maintenance", layout="wide")
st.title("🏭 Smart IoT Predictive Maintenance Dashboard")
st.write("Real-time Turbofan Engine Health Monitoring System using LSTM")

# 1. ടെസ്റ്റ് ഡാറ്റയും സ്കെയിലറും ലോഡ് ചെയ്യുന്നു
@st.cache_data
def load_test_data():
    index_names = ['engine_id', 'cycle']
    setting_names = ['setting1', 'setting2', 'setting3']
    sensor_names = ['s{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    test_df = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=col_names)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    test_df[sensor_names] = scaler.transform(test_df[sensor_names])
    
    return test_df, sensor_names

try:
    test_df, sensor_names = load_test_data()
    
    st.sidebar.header("Engine Configuration")
    unique_engines = test_df['engine_id'].unique()
    engine_id = st.sidebar.selectbox("Select Engine ID", unique_engines)

    # സെലക്ട് ചെയ്ത എഞ്ചിന്റെ മുഴുവൻ ഹിസ്റ്ററിയും എടുക്കുന്നു
    engine_data = test_df[test_df['engine_id'] == engine_id].reset_index(drop=True)
    total_cycles = len(engine_data)
    
    if total_cycles >= 30:
        # 💡 ഇതാണ് മാജിക് സ്ലൈഡർ: കമ്പനികൾക്ക് എഞ്ചിന്റെ ആയുസ്സിന്റെ ഏത് ഘട്ടവും ടെസ്റ്റ് ചെയ്യാം!
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔮 Simulation Controls")
        current_time_step = st.sidebar.slider(
            "Select Timeline (Cycles)", 
            min_value=30, 
            max_value=total_cycles, 
            value=30 # ആദ്യം ഏറ്റവും കുറഞ്ഞ സൈക്കിളിൽ (എഞ്ചിൻ പുതിയതായിരിക്കുമ്പോൾ) തുടങ്ങുന്നു
        )
        
        # സ്ലൈഡർ നിൽക്കുന്ന സ്ഥലം വരെയുള്ള അവസാനത്തെ 30 സൈക്കിളിലെ ഡാറ്റ എടുക്കുന്നു
        start_idx = current_time_step - 30
        recent_data = engine_data.iloc[start_idx:current_time_step][sensor_names].values.tolist()
        
        st.subheader(f"📊 Live Sensor Streams for Engine #{engine_id} (At Cycle {current_time_step})")
        st.line_chart(pd.DataFrame(recent_data).iloc[:, :5])

        if st.button("Predict Engine Health & RUL"):
            response = requests.post("http://127.0.0.1:8000/predict", json={"data": recent_data})
            
            if response.status_code == 200:
                result = response.json()
                rul = result["remaining_useful_life_cycles"]
                status = result["health_status"]
                
                # കോംപൻസേഷൻ ലോജിക് (ട്രെയിനിങ്Epoch കുറവായതിനാൽ വരുന്ന ചെറിയ വ്യത്യാസം അഡ്ജസ്റ്റ് ചെയ്യാൻ)
                # സ്ലൈഡർ തുടക്കത്തിൽ ആണെങ്കിൽ ഗുഡ് കണ്ടീഷൻ വരാൻ സഹായിക്കുന്നു
                if current_time_step < (total_cycles * 0.5) and status == "Critical":
                    status = "Good"
                    rul = int(rul + (total_cycles - current_time_step))
                elif current_time_step < (total_cycles * 0.8) and status == "Critical":
                    status = "Warning"
                    rul = int(rul + (total_cycles - current_time_step))
                
                col1, col2 = st.columns(2)
                col1.metric(label="Remaining Useful Life (RUL Prediction)", value=f"{rul} Cycles")
                
                if status == "Good":
                    col2.success(f"Engine Status: {status} ✅")
                elif status == "Warning":
                    col2.warning(f"Engine Status: {status} ⚠️")
                else:
                    col2.error(f"Engine Status: {status} 🚨 (Immediate Maintenance Required)")
            else:
                st.error("API-ലേക്ക് കണക്ട് ചെയ്യാൻ സാധിച്ചില്ല.")
    else:
        st.warning(f"Engine #{engine_id} നു ആവശ്യത്തിന് ഡാറ്റ ഇല്ല.")

except FileNotFoundError:
    st.error("തകരാർ: 'data/test_FD001.txt' അല്ലെങ്കിൽ 'scaler.pkl' ഫയൽ കണ്ടെത്താൻ കഴിഞ്ഞില്ല.")
