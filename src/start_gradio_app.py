import gradio as gr
from sampling_functions import sample

title = "Amigo's Brains with Diffusion Models"
description = """
Test our diffusion models trained to generate brain imaging data!
"""

article = """
By Walter Hugo Lopez Pinaya from [AMIGO](https://amigos.ai/)
<center><img src="https://amigos.ai/assets/images/logo_dark_rect.png" width=300px></center>
"""

inputs_block = [
    gr.Radio(
        choices=["Female", "Male"],
        value="Female",
        type="index",
        label="Gender"
    ),
    gr.Slider(
        minimum=44,
        maximum=82,
        value=63,
        label="Age [years]",
    ),
    gr.Slider(
        minimum=0,
        maximum=1,
        value=0.5,
        label="Volume of ventricular cerebrospinal fluid",
    ),
    gr.Slider(
        minimum=0,
        maximum=1,
        value=0.5,
        label="Volume of brain, grey+white matter (normalised for head size)",
    ),
]

model = None


demo = gr.Interface(
    sample,
    inputs_block,
    "number",
    title=title,
    description=description,
    article=article,
    flagging_dir="/media/walter/Storage/Projects/gradio_medical_ldm/src/flagged"
)

demo.launch()

