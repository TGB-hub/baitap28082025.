# baitap28082025.
!pip -q install gradio==4.*


import gradio as gr
import numpy as np


ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}
LABEL2COLOR = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}


TYPE_SCORE   = {0: 0.20, 1: 0.60, 2: 1.00}   # Normal < Fragile < High value
SPEED_SCORE  = {0: 0.30, 1: 0.60, 2: 1.00}   # Economy < Standard < Express
PREF_SCORE   = {0: 0.20, 1: 0.60, 2: 1.00}   # Cost-first < Balanced < Speed-first

def compute_priority(weight, distance, type_goods, speed, urgency_hours, customer_pref,
                     w_weight, w_dist, w_type, w_speed, w_urgency, w_pref):
   
    w = np.array([w_weight, w_dist, w_type, w_speed, w_urgency, w_pref], dtype=float)
    w = w / max(w.sum(), 1e-9)

    
    weight_n   = min(max(weight   / 120.0, 0), 1)     # 0..120 kg
    distance_n = min(max(distance / 400.0, 0), 1)     # 0..400 km
    type_n     = TYPE_SCORE[int(type_goods)]
    speed_n    = SPEED_SCORE[int(speed)]
 
    urgency_n  = min(max((72.0 - urgency_hours) / 72.0, 0), 1)
    pref_n     = PREF_SCORE[int(customer_pref)]

    feats = np.array([weight_n, distance_n, type_n, speed_n, urgency_n, pref_n])

    score = float((w * feats).sum())

  
    if score < 0.40:
        pid = 0
    elif score < 0.70:
        pid = 1
    else:
        pid = 2

    label = ID2LABEL[pid]
    color = LABEL2COLOR[label]


    chip = f"""
    <div style="display:inline-block;padding:.65rem 1.1rem;border-radius:9999px;
                background:{color};color:white;font-weight:800;letter-spacing:.6px">
      PRIORITY: {label}
    </div>"""

    expl = f"""
    <div style="margin-top:.75rem; font-size:14px; color:#334155">
      <b>Inputs</b> — Weight: {weight:.0f} kg • Distance: {distance:.0f} km • Type: {int(type_goods)}
      • Speed: {int(speed)} • Urgency: {int(urgency_hours)}h • Pref: {int(customer_pref)}
      <br><b>Normalized features</b> — W:{weight_n:.2f}, D:{distance_n:.2f}, T:{type_n:.2f},
      S:{speed_n:.2f}, U:{urgency_n:.2f}, P:{pref_n:.2f}
      <br><b>Weighted score</b>: {score:.2f}
    </div>"""

    return chip + expl, pid

theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(theme=theme, css="""
.gradio-container {background: linear-gradient(180deg,#f8fafc,#ffffff);}
#card {max-width: 860px; margin: 0 auto;}
.title {font-weight: 900; letter-spacing: .4px;}
.subtitle {color:#475569}
""") as demo:
    with gr.Column(elem_id="card"):
        gr.Markdown("<h1 class='title'>Logistics Priority Estimator</h1>"
                    "<p class='subtitle'>Chọn tham số bằng thanh trượt. Kết quả là "
                    "<b>mức độ ưu tiên</b> (Low / Medium / High).</p>")

        with gr.Group():
            weight   = gr.Slider(0, 120, value=20, step=1, label="Order size (kg)")
            distance = gr.Slider(0, 400, value=80, step=5, label="Distance (km)")
            type_g   = gr.Slider(0, 2, value=0, step=1, label="Type of goods (0=Normal, 1=Fragile, 2=High value)")
            speed    = gr.Slider(0, 2, value=1, step=1, label="Delivery speed (0=Economy, 1=Standard, 2=Express)")
            urgency  = gr.Slider(0, 72, value=24, step=1, label="Delivery urgency (hours to deadline; lower = more urgent)")
            pref     = gr.Slider(0, 2, value=1, step=1, label="Customer preference (0=Cost, 1=Balanced, 2=Speed)")

        with gr.Accordion("⚙️ Advanced: Weight tuning (tổng sẽ được chuẩn hoá thành 1.0)", open=False):
            w_weight  = gr.Slider(0, 1, 0.20, step=0.01, label="Weight importance")
            w_dist    = gr.Slider(0, 1, 0.20, step=0.01, label="Distance importance")
            w_type    = gr.Slider(0, 1, 0.25, step=0.01, label="Type importance")
            w_speed   = gr.Slider(0, 1, 0.15, step=0.01, label="Speed importance")
            w_urgency = gr.Slider(0, 1, 0.15, step=0.01, label="Urgency importance")
            w_pref    = gr.Slider(0, 1, 0.05, step=0.01, label="Customer preference importance")

        btn = gr.Button("Predict Priority", variant="primary")
        out_html = gr.HTML()
        out_id   = gr.Number(label="Priority ID (0=Low, 1=Medium, 2=High)", interactive=False)

        btn.click(
            compute_priority,
            inputs=[weight, distance, type_g, speed, urgency, pref,
                    w_weight, w_dist, w_type, w_speed, w_urgency, w_pref],
            outputs=[out_html, out_id]
        )

demo.launch()
