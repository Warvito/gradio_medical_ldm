import shutil

import cv2
import gradio as gr
import mediapy
import mlflow.pytorch
import numpy as np
import torch
from skimage import img_as_ubyte

from models.ddim import DDIMSampler

ffmpeg_path = shutil.which('ffmpeg')
mediapy.set_ffmpeg(ffmpeg_path)

# Loading model
vqvae = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative_modelling_ldm/mlruns/2/2f37b3b604a44b189b020028aa53f991/artifacts/final_model"
)
vqvae.eval()

diffusion = mlflow.pytorch.load_model(
    "/media/walter/Storage/Projects/generative_modelling_ldm/mlruns/6/c7b62c88595843d3a404368c87df5607/artifacts/final_model"
)
diffusion.eval()

device = torch.device("cuda")
diffusion = diffusion.to(device)
vqvae = vqvae.to(device)


def sample_fn(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
):
    print("Sampling brain!")
    print(f"Gender: {gender_radio}")
    print(f"Age: {age_slider}")
    print(f"Ventricular volume: {ventricular_slider}")
    print(f"Brain volume: {brain_slider}")

    age_slider = (age_slider - 44) / (82 - 44)

    cond = torch.Tensor([[gender_radio, age_slider, ventricular_slider, brain_slider]])
    latent_shape = [1, 3, 20, 28, 20]
    cond_crossatten = cond.unsqueeze(1)
    cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(latent_shape[2:]))
    conditioning = {
        'c_concat': [cond_concat.float().to(device)],
        'c_crossattn': [cond_crossatten.float().to(device)],
    }

    ddim = DDIMSampler(diffusion)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(latent_shape[1:]),
        eta=1.0
    )

    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors).cpu()

    return x_hat.numpy()


def create_videos(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
):
    image_data = sample_fn(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
    )
    image_data = image_data[0, 0, 5:-5, 5:-5, :-15]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

    # Write frames to video
    with mediapy.VideoWriter("./brain_axial.mp4", shape=(150, 214), fps=12, codec="vp9") as w:
        for idx in range(image_data.shape[2]):
            img = (image_data[:, :, idx] * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    # Write frames to video
    with mediapy.VideoWriter("./brain_sagittal.mp4", shape=(145, 214), fps=12, codec="vp9") as w:
        for idx in range(image_data.shape[0]):
            img = (np.rot90(image_data[idx, :, :]) * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    # Write frames to video
    with mediapy.VideoWriter("./brain_coronal.mp4", shape=(145, 150), fps=12, codec="vp9") as w:
        # with mediapy.VideoWriter("./brain_coronal.mp4", shape=(150, 145), fps=12, codec="vp9") as w:
        for idx in range(image_data.shape[1]):
            img = (np.rot90(np.flip(image_data, axis=1)[:, idx, :]) * 255).astype(np.uint8)
            # img = (image_data[:, idx, :] * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    return "./brain_axial.mp4", "./brain_sagittal.mp4", "./brain_coronal.mp4"


# TEXT
title = "Generating Brain Imaging with Diffusion Models"
description = """
<center>Gradio demo for our brain generator 🧠</center>
<center><a href="https://amigos.ai/">[PAPER]</a> <a href="https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b">[DATASET]</a></center>
"""

article = """
Checkout our dataset with [100K synthetic brain](https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b)! 🧠🧠🧠

App made by [Walter Hugo Lopez Pinaya](https://twitter.com/warvito) from [AMIGO](https://amigos.ai/)
<center><img src="https://amigos.ai/assets/images/logo_dark_rect.png" alt="amigos.ai" width=300px></center>
"""

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>"
        + title
        + "</h1>"
    )
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Inputs"):
                        with gr.Row():
                            gr.Markdown("Choose how your generated brain will look like")
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
                                label="Volume of brain",
                                interactive=True,
                            )
                        with gr.Row():
                            submit_btn = gr.Button("Generate", variant="primary")

                    with gr.TabItem("Unrestricted Inputs"):
                        with gr.Row():
                            gr.Markdown("Be free to use any value to generate the wildest brains!")
                        with gr.Row():
                            unrest_gender_number = gr.Number(
                                value=1.0,
                                precision=1,
                                label="Gender [Female=0, Male=1]",
                                interactive=True,
                            )
                            unrest_age_number = gr.Number(
                                value=63,
                                precision=1,
                                label="Age [years]",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_ventricular_number = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of ventricular cerebrospinal fluid",
                                interactive=True,
                            )
                            unrest_brain_number = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of brain",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_submit_btn = gr.Button("Generate", variant="primary")

                        gr.Examples(
                            examples=[
                                [1, 63, 1.3, 0.5],
                                [0, 63, 1.9, 0.5],
                                [1, 63, -0.5, 0.5],
                                [0, 63, 0.5, -0.3],
                            ],
                            inputs=[
                                unrest_gender_number,
                                unrest_age_number,
                                unrest_ventricular_number,
                                unrest_brain_number,
                            ]
                        )

        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Axial View"):
                        axial_sample_plot = gr.Video(show_label=False)
                    with gr.TabItem("Sagittal View"):
                        sagittal_sample_plot = gr.Video(show_label=False)
                    with gr.TabItem("Coronal View"):
                        coronal_sample_plot = gr.Video(show_label=False)
    gr.Markdown(article)

    submit_btn.click(
        create_videos,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )
    unrest_submit_btn.click(
        create_videos,
        [
            unrest_gender_number,
            unrest_age_number,
            unrest_ventricular_number,
            unrest_brain_number,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )

demo.launch(share=True)
