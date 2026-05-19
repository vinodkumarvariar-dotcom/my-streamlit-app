import streamlit as st

st.title(" I am vinod kumar @ streamlit")

st.header("Automated Arrhythmia Detection from ECG Signals Using Convolutional Neural Networks")
#streamlit run main.py (paste and run at terminal for see streamlit file)
#streamlit run file_name.py (paste and run at terminal for see streamlit file)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# 1. പേജ് കോൺഫിഗറേഷൻ (Page Styling)
st.set_page_config(
    page_title="Automated Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)

# ഹെഡ്ഡർ ഡിസൈൻ
st.title("❤️ Automated Arrhythmia Detection from ECG Signals")
st.write("Convolutional Neural Networks (CNN) ഉപയോഗിച്ചുള്ള കൃത്യമായ ഇസിജി തരംഗ വർഗ്ഗീകരണം.")
st.markdown("---")

# 2. ഡമ്മി മോഡൽ ലോഡിംഗ് / ക്രിയേഷൻ (ഫോർ ടെസ്റ്റിംഗ്)
@st.cache_resource
def load_ecg_model():
    model_path = 'ecg_cnn_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # സിസ്റ്റത്തിൽ മോഡൽ ഫയൽ ഇല്ലെങ്കിൽ താൽക്കാലികമായി ഒരു ഡമ്മി മോഡൽ ഉണ്ടാക്കുന്നു
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(186, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

model = load_ecg_model()
classes = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

# 3. സൈഡ്‌ബാർ ഇന്റർഫേസ് (Sidebar Options)
st.sidebar.header("📋 കൺട്രോൾ പാനൽ")
app_mode = st.sidebar.selectbox("ഒരു ഓപ്ഷൻ തിരഞ്ഞെടുക്കുക:", ["ഹോം പേജ്", "ഇസിജി പ്രവചനം (Prediction)"])

if app_mode == "ഹോം പേജ്":
    st.subheader("📌 പ്രൊജക്റ്റിനെക്കുറിച്ച്")
    st.write("""
    ഈ ആപ്ലിക്കേഷൻ 1D-CNN (Convolutional Neural Network) മോഡൽ ഉപയോഗിച്ച് പ്രവർത്തിക്കുന്നു. 
    ഇസിജി സിഗ്നലുകളിലെ ചെറിയ വ്യതിയാനങ്ങൾ പോലും തിരിച്ചറിഞ്ഞ് താഴെ പറയുന്ന അഞ്ച് തരം ഹൃദയമിടിപ്പുകളെ വർഗ്ഗീകരിക്കാൻ ഇതിന് സാധിക്കും:
    - **Normal (N)**: സാധാരണ ഹൃദയമിടിപ്പ്
    - **Supraventricular (S)**: സുപ്രാവെൻട്രിക്കുലാർ അരിത്മിയ
    - **Ventricular (V)**: വെൻട്രിക്കുലാർ പ്രീമച്യുർ ബീറ്റ്
    - **Fusion (F)**: ഫ്യൂഷൻ ബീറ്റ്
    - **Unknown (Q)**: തിരിച്ചറിയാത്ത മറ്റ് തരംഗങ്ങൾ
    """)
    
    # മാതൃകയായി കാണിക്കാൻ ഒരു ഡമ്മി ഇസിജി ഗ്രാഫ്
    st.subheader("📊 മാതൃകാ ഇസിജി തരംഗം (Sample Waveform)")
    sample_t = np.linspace(0, 10, 186)
    sample_sig = np.sin(sample_t) + 0.5 * np.sin(2 * sample_t)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sample_sig, color='#FF4B4B', linewidth=2)
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

elif app_mode == "ഇസിജി പ്രവചനം (Prediction)":
    st.subheader("🔍 ഇസിജി സിഗ്നൽ പരിശോധന")
    
    # ഫയൽ അപ്‌ലോഡ് ചെയ്യാനുള്ള ഓപ്ഷൻ
    uploaded_file = st.file_uploader("നിങ്ങളുടെ ECG CSV ഫയൽ അപ്‌ലോഡ് ചെയ്യുക (186 ഡാറ്റാ പോയിന്റുകൾ അടങ്ങിയത്)", type=["csv"])
    
    # ടെസ്റ്റിംഗിനായി ഒരു ഡമ്മി ഫയൽ ഉണ്ടാക്കാനുള്ള ബട്ടൺ
    if st.button("💡 ഒരു മാതൃകാ ഡാറ്റ നിർമ്മിച്ച് പരിശോധിക്കുക"):
        random_signal = np.sin(np.linspace(0, 10, 186)) + np.random.normal(0, 0.1, 186)
        df_demo = pd.DataFrame(random_signal)
        df_demo.to_csv("sample_ecg.csv", index=False, header=False)
        st.success("`sample_ecg.csv` എന്ന ഫയൽ തനിയെ നിർമ്മിക്കപ്പെട്ടു! ഇത് മുകളിൽ അപ്‌ലോഡ് ചെയ്ത് പരീക്ഷിക്കാം.")

    if uploaded_file is not None:
        try:
            # ഫയൽ റീഡ് ചെയ്യുന്നു
            df = pd.read_csv(uploaded_file, header=None)
            ecg_signal = df.values.flatten()
            
            if len(ecg_signal) != 186:
                st.error(f"❌ തെറ്റായ ഡാറ്റാ വലിപ്പം! ഫയലിൽ കൃത്യം 186 പോയിന്റുകൾ ഉണ്ടായിരിക്കണം. (ഇപ്പോൾ ഉള്ളത്: {len(ecg_signal)})")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.success("✅ ഫയൽ വിജയകരമായി അപ്‌ലോഡ് ചെയ്തു!")
                    # ഇസിജി ഗ്രാഫ് വരയ്ക്കുന്നു
                    st.subheader("📈 അപ്‌ലോഡ് ചെയ്ത ഇസിജി തരംഗം")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(ecg_signal, color='#1E88E5', linewidth=2)
                    ax.set_title("ECG Waveform Visualizer")
                    ax.grid(True)
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("🤖 CNN മോഡൽ പ്രവചനം")
                    with st.spinner('സിഗ്നൽ വിശകലനം ചെയ്തുകൊണ്ടിരിക്കുന്നു...'):
                        # പ്രീപ്രോസസ്സിംഗ് (റീഷെയ്പ്പിംഗ്)
                        input_data = ecg_signal.reshape(1, 186, 1)
                        # പ്രവചനം നടത്തുന്നു
                        prediction = model.predict(input_data)[0]
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[predicted_class] * 100
                        
                    # ഫലം സ്ക്രീനിൽ കാണിക്കുന്നു
                    st.metric(label="കണ്ടെത്തിയ രോഗാവസ്ഥ (Result)", value=classes[predicted_class])
                    st.metric(label="കൃത്യത സാധ്യത (Confidence)", value=f"{confidence:.2f} %")
                    
                    # ഓരോ ക്ലാസിന്റെയും സാധ്യതകൾ കാണിക്കുന്ന ഒരു ചെറിയ പ്രോഗ്രസ് ബാർ
                    st.write("📊 **സാധ്യതകളുടെ പട്ടിക:**")
                    for i, cls in enumerate(classes):
                        st.write(f"{cls}")
                        st.progress(float(prediction[i]))
                        
        except Exception as e:
            st.error(f"ഫയൽ പ്രോസസ്സ് ചെയ്യുന്നതിൽ തകരാർ സംഭവിച്ചു: {e}")

           