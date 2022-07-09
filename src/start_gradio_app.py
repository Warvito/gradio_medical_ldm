import gradio as gr
import nibabel as nib
import time
from ploty_image import draw
model = None
image_data = None

def sample_fn(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
):
    print(gender_radio)
    print(age_slider)
    print(ventricular_slider)
    print(brain_slider)
    image = nib.load("/home/walter/Downloads/sub-HCD0001305_ses-01_space-MNI152NLin2009aSym_T2w.nii.gz")
    global image_data
    image_data = image.get_fdata()
    time.sleep(2)

    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    # plotted_img = image_data[:, :, 80]
    # plotted_img = (plotted_img - plotted_img.min()) / (plotted_img.max() - plotted_img.min())
    return [
        image_data[:, :, 10],
        image_data[:, :, 20],
        image_data[:, :, 30],
        image_data[:, :, 40],
        image_data[:, :, 50],
        image_data[:, :, 60],
        image_data[:, :, 70],
    ]

title = "Amigo's Brains with Diffusion Models"
description = """
Test our diffusion models trained to generate brain imaging data!
"""

article = """
By Walter Hugo Lopez Pinaya from [AMIGO](https://amigos.ai/)
<center><img src="https://amigos.ai/assets/images/logo_dark_rect.png" width=300px></center>
"""

import plotly.express as px


def get_fig(a,b,c,d):
    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
                 animation_frame="year", animation_group="country", range_y=[0, 4000000000])
    return fig

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + title
        + "</h1>"
    )
    gr.Markdown(description)
    with gr.Box():
        gr.Markdown("<h2>Inputs<h2>")
        with gr.Row():
            gender_radio = gr.Radio(
                choices=["Female", "Male"],
                value="Female",
                type="index",
                label="Gender",
                interactive=True,
            )
            age_slider = gr.Slider(
                minimum=44,
                maximum=82,
                value=63,
                label="Age [years]",
                interactive=True,
            )
        with gr.Row():
            ventricular_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                label="Volume of ventricular cerebrospinal fluid",
                interactive=True,
            )
            brain_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                label="Volume of brain, grey+white matter (normalised for head size)",
                interactive=True,
            )
    with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit", variant="primary")

    # with gr.Box():
    #     with gr.Row():
    #         axis_radio = gr.Radio(
    #             choices=["Sagittal", "Coronal", "Axial"],
    #             value="Axial",
    #             type="index",
    #             show_label=False,
    #             interactive=True,
    #         )
    sample_plot = gr.Plot()
    gr.Markdown(article)

    # clear_btn.click()


    submit_btn.click(
        get_fig,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        sample_plot,
    )


demo.launch()
