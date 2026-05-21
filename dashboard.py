#python dashboard.py 
# (paste and run at terminal and get a url, pasteb this url in to bowser)
#http://127.0.0.1:8050/


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# 1. നാല് വ്യത്യസ്ത ഡാറ്റാബേസുകളുടെ ഡാറ്റാ ലോഡിംഗ് / സിമുലേഷൻ
# ==============================================================================
np.random.seed(7)
db_names = [
    "MIT-BIH Arrhythmia Database",
    "MIT-BIH Supraventricular Arrhythmia Database",
    "INCART 2-lead Arrhythmia Database",
    "Sudden Cardiac Death Holter Database"
]

classes = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
db_data = {}

# ഓരോ ഡാറ്റാബേസിനും അനുയോജ്യമായ രീതിയിൽ ഡാറ്റ സജ്ജീകരിക്കുന്നു
for db in db_names:
    num_samples = 150
    time_points = 187 # സ്റ്റാൻഡേർഡ് ECG വിൻഡോ ലെങ്ത്
    
    # യഥാർത്ഥ CSV ഫയൽ ലോഡ് ചെയ്യാൻ താഴെയുള്ള വരികൾ ഉപയോഗിക്കാം:
    # if db == "MIT-BIH Arrhythmia Database": df_raw = pd.read_csv('mitbih_arr.csv', header=None)
    
    # സിഗ്നൽ വെയ്‌വുകൾ ജനറേറ്റ് ചെയ്യുന്നു
    ecg_signals = np.random.randn(num_samples, time_points) * 0.12
    for i in range(num_samples):
        ecg_signals[i, 35:55] += np.random.uniform(0.3, 0.5)   # P Wave
        ecg_signals[i, 75:90] += np.random.uniform(1.3, 1.8)   # QRS Complex
        ecg_signals[i, 115:135] += np.random.uniform(0.4, 0.7) # T Wave
        
    # ഡാറ്റാബേസിന്റെ സ്വഭാവത്തിനനുസരിച്ച് ക്ലാസ് പ്രോബബിലിറ്റി മാറ്റുന്നു
    if "Supraventricular" in db:
        p_dist = [0.4, 0.4, 0.1, 0.05, 0.05] # S-Type കൂടുതൽ
    elif "Death" in db:
        p_dist = [0.3, 0.1, 0.5, 0.05, 0.05] # V-Type (മാരകമായവ) കൂടുതൽ
    else:
        p_dist = [0.7, 0.1, 0.1, 0.05, 0.05] # Normal കൂടുതൽ

    true_lbls = np.random.choice(list(classes.keys()), size=num_samples, p=p_dist)
    
    # മോഡൽ പ്രെഡിക്ഷൻ (88% - 94% ആക്യുറസി സിമുലേഷൻ)
    acc_rate = np.random.uniform(0.88, 0.94)
    pred_lbls = np.array([
        t if np.random.rand() < acc_rate else np.random.choice(list(classes.keys()))
        for t in true_lbls
    ])
    
    df = pd.DataFrame(ecg_signals)
    df['True_Label'] = [classes[x] for x in true_lbls]
    df['Pred_Label'] = [classes[x] for x in pred_lbls]
    df['Sample_ID'] = [f"Rec_{100+i}" for i in range(num_samples)]
    
    db_data[db] = df

# ==============================================================================
# 2. ഡാഷ്‌ബോർഡ് ലേഔട്ട് ഡിസൈൻ
# ==============================================================================
app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Segoe UI, Arial, sans-serif', 'padding': '25px', 'backgroundColor': '#f8fafc'}, children=[
    
    # ഹെഡർ പാനൽ
    html.Div(style={'backgroundColor': '#0f172a', 'color': 'white', 'padding': '25px', 'borderRadius': '12px', 'marginBottom': '25px', 'boxShadow': '0 4px 6px -1px rgb(0 0 0 / 0.1)'}, children=[
        html.H1("Multi-Database ECG Arrhythmia Classification Hub", style={'margin': '0', 'fontSize': '28px', 'fontWeight': '700'}),
        html.P("Comparative Machine Learning Analytics for MIT-BIH, INCART, and SCD Holter Databases", style={'margin': '8px 0 0 0', 'opacity': '0.8', 'fontSize': '15px'}),
        
        # ഡാറ്റാബേസ് സെലക്ടർ ഡ്രോപ്പ്ഡൗൺ (ഇവിടെയാണ് മെയിൻ ഫിൽട്ടറിംഗ്)
        html.Div(style={'marginTop': '20px', 'maxWidth': '500px'}, children=[
            html.Label("Choose Active ECG Database:", style={'fontWeight': '600', 'color': '#94a3b8', 'display': 'block', 'marginBottom': '8px'}),
            dcc.Dropdown(
                id='database-selector',
                options=[{'label': db, 'value': db} for db in db_names],
                value=db_names[0],
                clearable=False,
                style={'color': '#0f172a', 'borderRadius': '6px'}
            )
        ])
    ]),
    
    # ഡൈനാമിക് KPI കാർഡുകൾ
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px'}, children=[
        html.Div(id='kpi-accuracy', style={'flex': '1', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'borderLeft': '6px solid #10b981'}),
        html.Div(id='kpi-total', style={'flex': '1', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'borderLeft': '6px solid #3b82f6'}),
        html.Div(id='kpi-anomaly', style={'flex': '1', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'borderLeft': '6px solid #ef4444'})
    ]),

    # ഗ്രാഫുകളും വ്യൂവറും അടങ്ങിയ പ്രധാന സെക്ഷൻ
    html.Div(style={'display': 'flex', 'gap': '25px', 'flexWrap': 'wrap'}, children=[
        
        # ഇടത് വശം: സ്റ്റാറ്റിസ്റ്റിക്സ്
        html.Div(style={'flex': '1', 'minWidth': '480px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}, children=[
            html.H3("Selected Database Distribution", style={'marginTop': '0', 'color': '#1e293b', 'fontSize': '18px'}),
            dcc.Graph(id='multi-dist-plot'),
            
            html.H3("Confusion Matrix Matrix", style={'color': '#1e293b', 'fontSize': '18px', 'marginTop': '20px'}),
            dcc.Graph(id='multi-cm-plot')
        ]),
        
        # വലത് വശം: ലൈവ് സിഗ്നൽ ഇൻസ്പെക്ടർ
        html.Div(style={'flex': '1.3', 'minWidth': '580px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '12px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'}, children=[
            html.H3("Interactive Patient Signal Inspector", style={'marginTop': '0', 'color': '#1e293b', 'fontSize': '18px'}),
            
            html.Div(style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    html.Label("Filter Arrhythmia Class:", style={'fontWeight': '600', 'fontSize': '13px', 'color': '#475569'}),
                    dcc.Dropdown(id='multi-class-dropdown', clearable=False)
                ]),
                html.Div(style={'flex': '1'}, children=[
                    html.Label("Select Sample Record:", style={'fontWeight': '600', 'fontSize': '13px', 'color': '#475569'}),
                    dcc.Dropdown(id='multi-patient-dropdown', clearable=False)
                ])
            ]),
            
            html.Div(id='multi-diagnostic-panel'),
            dcc.Graph(id='multi-signal-plot')
        ])
    ])
])

# ==============================================================================
# 3. ഇന്ററാക്ടീവ് ഫങ്ഷനുകൾ (Callbacks)
# ==============================================================================

# ഡാറ്റാബേസ് മാറുമ്പോൾ ലഭ്യമായ ക്ലാസുകൾ അപ്‌ഡേറ്റ് ചെയ്യാൻ
@app.callback(
    Output('multi-class-dropdown', 'options'),
    Output('multi-class-dropdown', 'value'),
    Input('database-selector', 'value')
)
def update_classes(selected_db):
    df = db_data[selected_db]
    unique_classes = sorted(list(df['True_Label'].unique()))
    options = [{'label': c, 'value': c} for c in unique_classes]
    return options, unique_classes[0]

# ക്ലാസ് മാറുമ്പോൾ രോഗികളുടെ ലിസ്റ്റ് (Sample ID) അപ്‌ഡേറ്റ് ചെയ്യാൻ
@app.callback(
    Output('multi-patient-dropdown', 'options'),
    Output('multi-patient-dropdown', 'value'),
    Input('database-selector', 'value'),
    Input('multi-class-dropdown', 'value')
)
def update_patients(selected_db, selected_class):
    df = db_data[selected_db]
    filtered = df[df['True_Label'] == selected_class]
    options = [{'label': sid, 'value': sid} for sid in filtered['Sample_ID']]
    return options, options[0]['value'] if options else None

# പ്രധാന ഗ്രാഫുകളും KPI-കളും അപ്‌ഡേറ്റ് ചെയ്യാൻ
@app.callback(
    Output('kpi-accuracy', 'children'),
    Output('kpi-total', 'children'),
    Output('kpi-anomaly', 'children'),
    Output('multi-dist-plot', 'figure'),
    Output('multi-cm-plot', 'figure'),
    Output('multi-signal-plot', 'figure'),
    Output('multi-diagnostic-panel', 'children'),
    Output('multi-diagnostic-panel', 'style'),
    Input('database-selector', 'value'),
    Input('multi-class-dropdown', 'value'),
    Input('multi-patient-dropdown', 'value')
)
def refresh_dashboard(selected_db, selected_class, selected_patient):
    df = db_data[selected_db]
    all_classes = sorted(list(df['True_Label'].unique()))
    
    # 1. KPI കണക്കുകൂട്ടലുകൾ
    report = classification_report(df['True_Label'], df['Pred_Label'], labels=all_classes, output_dict=True)
    acc = report['accuracy'] * 100
    total_beats = len(df)
    abnormal_count = len(df[df['True_Label'] != 'Normal (N)'])
    
    kpi_acc = [html.H4("Classification Accuracy", style={'margin':'0','color':'#64748b','fontSize':'13px'}), html.H2(f"{acc:.2f}%", style={'margin':'5px 0 0 0','color':'#0f172a'})]
    kpi_tot = [html.H4("Total Waveforms Analyzed", style={'margin':'0','color':'#64748b','fontSize':'13px'}), html.H2(f"{total_beats} Beats", style={'margin':'5px 0 0 0','color':'#0f172a'})]
    kpi_ano = [html.H4("Total Arrhythmia Load", style={'margin':'0','color':'#64748b','fontSize':'13px'}), html.H2(f"{abnormal_count} Cases", style={'margin':'5px 0 0 0','color':'#0f172a'})]

    # 2. ചാർട്ട് 1: ക്ലാസ് ഡിസ്ട്രിബ്യൂഷൻ
    counts = df['True_Label'].value_counts()
    fig_dist = go.Figure(data=[go.Bar(x=counts.index, y=counts.values, marker_color='#3b82f6', text=counts.values, textposition='auto')])
    fig_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=200, plot_bgcolor='rgba(0,0,0,0)', yaxis={'visible': False})

    # 3. ചാർട്ട് 2: കൺഫ്യൂഷൻ മാട്രിക്സ്
    cm = confusion_matrix(df['True_Label'], df['Pred_Label'], labels=all_classes)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=all_classes, y=all_classes, colorscale='Reds', showscale=False, text=cm, texttemplate="%{text}"))
    fig_cm.update_layout(margin=dict(l=40, r=10, t=10, b=40), height=230, xaxis_title="Model Predicted", yaxis_title="Physician Ground Truth")

    # 4. ചാർട്ട് 3: സിഗ്നൽ ഗ്രാഫും സ്റ്റാറ്റസ് പാനലും
    if selected_patient:
        row = df[df['Sample_ID'] == selected_patient].iloc[0]
        # അവസാന രണ്ട് കോളങ്ങൾ (True_Label, Pred_Label) ഒഴിവാക്കി ബാക്കി സിഗ്നൽ വാല്യൂസ് എടുക്കുന്നു
        amplitudes = row.iloc[:-3].values 
        t_label = row['True_Label']
        p_label = row['Pred_Label']
        
        fig_signal = go.Figure(data=go.Scatter(y=amplitudes, mode='lines', line=dict(color='#dc2626', width=2.5)))
        fig_signal.update_layout(
            xaxis_title="Time Sample Steps (187 Point Window)", yaxis_title="Voltage Amplitude (mV)",
            margin=dict(l=40, r=10, t=10, b=40), height=280, plot_bgcolor='#f8fafc'
        )
        
        match = (t_label == p_label)
        bg = '#d1fae5' if match else '#fee2e2'
        tx = '#065f46' if match else '#991b1b'
        
        panel_content = [
            html.Span(f"Database: {selected_db} | ID: {selected_patient}"),
            html.Span(f"Truth: {t_label} ➔ Predicted: {p_label} ({'✅ Match' if match else '❌ Alert'})")
        ]
        panel_style = {'padding': '12px', 'borderRadius': '8px', 'marginBottom': '15px', 'fontWeight': '600', 'backgroundColor': bg, 'color': tx, 'display': 'flex', 'justifyContent': 'space-between', 'fontSize':'14px'}
    else:
        fig_signal = go.Figure()
        panel_content = ["Select parameters to view waveform"]
        panel_style = {}

    return kpi_acc, kpi_tot, kpi_ano, fig_dist, fig_cm, fig_signal, panel_content, panel_style

if __name__ == '__main__':
    app.run(debug=True)