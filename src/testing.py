import numpy as np
import gradio as gr

demo = gr.Blocks()

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with demo:
    gr.Markdown("Flip text or image files using this demo.")
    text_input = gr.Textbox()
    text_button = gr.Button("Flip")
    text_output = gr.Textbox()

    text_button.click(flip_text, inputs=text_input, outputs=text_output)

demo.launch()