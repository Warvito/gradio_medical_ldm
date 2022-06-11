import gradio as gr
import nibabel as nib
import time

model = None
image_data = None

def sample_fn(
        seed,
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
):
    print(seed)
    print(gender_radio)
    print(age_slider)
    print(ventricular_slider)
    print(brain_slider)
    image = nib.load("/home/walter/Downloads/sub-HCD0001305_ses-01_space-MNI152NLin2009aSym_T2w.nii.gz")
    global image_data
    image_data = image.get_fdata()
    time.sleep(2)

    plotted_img = image_data[:, :, 80]
    plotted_img = (plotted_img - plotted_img.min()) / (plotted_img.max() - plotted_img.min())
    return plotted_img

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

def update_plotted_img(idx):
    global image_data
    print(idx)
    plotted_img = image_data[:, :, idx]
    plotted_img = (plotted_img - plotted_img.min()) / (plotted_img.max() - plotted_img.min())

    return plotted_img


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
            seed = gr.Number(
                value=42,
                label="Seed",
                interactive=True,
            )
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

    with gr.Box():
        with gr.Row():
            axis_radio = gr.Radio(
                choices=["Sagittal", "Coronal", "Axial"],
                value="Axial",
                type="index",
                show_label=False,
                interactive=True,
            )
            axis_slider = gr.Slider(
                minimum=0,
                maximum=159,
                value=80,
                step=1,
                show_label=False,
                interactive=True,
            )
        sample_img = gr.Image(
            image_mode="L",
            type="numpy",
            interactive=False,
        )
    gr.Markdown(article)

    # clear_btn.click()

    submit_btn.click(
        sample_fn,
        [
            seed,
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        sample_img,
    )

    axis_slider.change(
        update_plotted_img,
        axis_slider,
        sample_img
    )

demo.launch(enable_queue=True)
