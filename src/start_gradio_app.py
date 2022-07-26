import random
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import mediapy
import mlflow.pytorch
import numpy as np
import torch
from skimage import img_as_ubyte

from models.ddim import DDIMSampler

# import nibabel as nib

ffmpeg_path = shutil.which("ffmpeg")
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
        "c_concat": [cond_concat.float().to(device)],
        "c_crossattn": [cond_crossatten.float().to(device)],
    }

    ddim = DDIMSampler(diffusion)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(latent_shape[1:]),
        eta=1.0,
    )

    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors).cpu()

    return x_hat.numpy()


def create_videos_and_file(
    gender_radio,
    age_slider,
    ventricular_slider,
    brain_slider,
):
    output_dir = Path(
        f"/media/walter/Storage/Projects/gradio_medical_ldm/outputs/{str(uuid.uuid4())}"
    )
    output_dir.mkdir(exist_ok=True)

    image_data = sample_fn(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
    )
    image_data = image_data[0, 0, 5:-5, 5:-5, :-15]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    image_data = (image_data * 255).astype(np.uint8)

    # Write frames to video
    out = cv2.VideoWriter(
        f"{str(output_dir)}/brain_axial.mp4",
        cv2.VideoWriter_fourcc(*'avc1'),
        12,
        (150, 214),
        False
    )
    for idx in range(image_data.shape[2]):
        img = image_data[:, :, idx]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frame = img_as_ubyte(img)
        out.write(frame)
    out.release()

    # with mediapy.VideoWriter(
    #     f"{str(output_dir)}/brain_axial.mp4", shape=(150, 214), fps=12, crf=18
    # ) as w:
    #     for idx in range(image_data.shape[2]):
    #         img = image_data[:, :, idx]
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         frame = img_as_ubyte(img)
    #         w.add_image(frame)

    # Write frames to video
    out = cv2.VideoWriter(
        f"{str(output_dir)}/brain_sagittal.mp4",
        cv2.VideoWriter_fourcc(*'avc1'),
        12,
        (145, 214)
    )
    for idx in range(image_data.shape[0]):
        img = np.rot90(image_data[idx, :, :])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frame = img_as_ubyte(img)
        out.write(frame)
    out.release()

    # with mediapy.VideoWriter(
    #     f"{str(output_dir)}/brain_sagittal.mp4", shape=(145, 214), fps=12, crf=18
    # ) as w:
    #     for idx in range(image_data.shape[0]):
    #         img = np.rot90(image_data[idx, :, :])
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         frame = img_as_ubyte(img)
    #         w.add_image(frame)

    # Write frames to video
    out = cv2.VideoWriter(
        f"{str(output_dir)}/brain_coronal.mp4",
        cv2.VideoWriter_fourcc(*'avc1'),
        12,
        (145, 150)
    )
    for idx in range(image_data.shape[1]):
        img = np.rot90(np.flip(image_data, axis=1)[:, idx, :])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        frame = img_as_ubyte(img)
        out.write(frame)
    out.release()

    # with mediapy.VideoWriter(
    #     f"{str(output_dir)}/brain_coronal.mp4", shape=(145, 150), fps=12, crf=18
    # ) as w:
    #     for idx in range(image_data.shape[1]):
    #         img = np.rot90(np.flip(image_data, axis=1)[:, idx, :])
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #         frame = img_as_ubyte(img)
    #         w.add_image(frame)

    # # Create file
    # affine = np.array(
    #     [
    #         [-1.0, 0.0, 0.0, 96.48149872],
    #         [0.0, 1.0, 0.0, -141.47715759],
    #         [0.0, 0.0, 1.0, -156.55375671],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    # empty_header = nib.Nifti1Header()
    # sample_nii = nib.Nifti1Image(image_data, affine, empty_header)
    # nib.save(sample_nii, f"{str(output_dir)}/my_brain.nii.gz")

    # time.sleep(2)

    return (
        f"{str(output_dir)}/brain_axial.mp4",
        f"{str(output_dir)}/brain_sagittal.mp4",
        f"{str(output_dir)}/brain_coronal.mp4",
        # f"{str(output_dir)}/my_brain.nii.gz",
    )


def randomise():
    random_age = round(random.uniform(44.0, 82.0), 2)
    return (
        random.choice(["Female", "Male"]),
        random_age,
        round(random.uniform(0, 1.0), 2),
        round(random.uniform(0, 1.0), 2),
    )


def unrest_randomise():
    random_age = round(random.uniform(18.0, 100.0), 2)
    return (
        random.choice([1, 0]),
        random_age,
        round(random.uniform(-1.0, 2.0), 2),
        round(random.uniform(-1.0, 2.0), 2),
    )


# TEXT
title = "Generating Brain Imaging with Diffusion Models"
description = """
<center><b>WORK IN PROGRESS. DO NOT SHARE.</b></center>
<center><a href="https://amigos.ai/">[PAPER]</a> <a href="https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b">[DATASET]</a></center>

<details>
<summary>Instructions</summary>

With this app, you can generate synthetic brain images with one click!<br />You have two ways to set how your generated brain will look like:<br />- Using the "Inputs" tab that creates well-behaved brains using the same value ranges that our models learned as described in paper linked above<br />- Or using the "Unrestricted Inputs" tab to generate the wildest brains!<br />After customisation, just hit "Generate" and wait a few seconds. You can also download your new brain and visualise it with your favorite nifti viewer app. <br />* Note: You might need to rename the downloaded file to "my_brain.nii.gz". <b>Enjoy!<b>
</details>

"""

article = """
Checkout our dataset with [100K synthetic brain](https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b)! 🧠🧠🧠

App made by [Walter Hugo Lopez Pinaya](https://twitter.com/warvito) from [AMIGO](https://amigos.ai/)
<center><img src="https://amigos.ai/assets/images/logo_dark_rect.png" alt="amigos.ai" width=300px></center>
"""

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>"
    )
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Inputs"):
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
                            randomize_btn = gr.Button("I'm Feeling Lucky")

                    with gr.TabItem("Unrestricted Inputs"):
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
                            unrest_randomize_btn = gr.Button("I'm Feeling Lucky")

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
                            ],
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
                # sample_file = gr.File(label="My Brain")

    gr.Markdown(article)

    submit_btn.click(
        create_videos_and_file,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        # [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot, sample_file],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )
    unrest_submit_btn.click(
        create_videos_and_file,
        [
            unrest_gender_number,
            unrest_age_number,
            unrest_ventricular_number,
            unrest_brain_number,
        ],
        # [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot, sample_file],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )

    randomize_btn.click(
        fn=randomise,
        inputs=[],
        queue=False,
        outputs=[gender_radio, age_slider, ventricular_slider, brain_slider],
    )

    unrest_randomize_btn.click(
        fn=unrest_randomise,
        inputs=[],
        queue=False,
        outputs=[
            unrest_gender_number,
            unrest_age_number,
            unrest_ventricular_number,
            unrest_brain_number,
        ],
    )

demo.launch(share=True, enable_queue=True, prevent_thread_lock=True)
# demo.launch(debug=True, enable_queue=True)
