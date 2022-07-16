import gradio as gr
import plotly.graph_objects as go
import numpy as np

def draw_plotly():
    fig = go.Figure()
    # Add traces, one for each slider step
    for step in np.arange(0, 5, 0.1):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=np.arange(0, 10, 0.01),
                y=np.sin(step * np.arange(0, 10, 0.01))))
    # Make 10th trace visible
    fig.data[10].visible = True
    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i)}],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders
    )
    return fig


demo = gr.Interface(fn=draw_plotly, inputs=[], outputs=gr.Plot())

demo.launch(server_name="0.0.0.0", server_port=7601)