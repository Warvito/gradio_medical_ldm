import random
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import mediapy
import nibabel as nib
import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from skimage import img_as_ubyte
from torch.cuda.amp import autocast

ffmpeg_path = shutil.which("ffmpeg")
mediapy.set_ffmpeg(ffmpeg_path)

# Loading model
aekl = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_channels=[64, 128, 128, 128],
    latent_channels=3,
    num_res_blocks=2,
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
aekl.load_state_dict(torch.load("./pretrained_models/autoencoder.pth"))
aekl.eval()


diffusion = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=7,
    out_channels=3,
    num_channels=[256, 512, 768],
    num_res_blocks=2,
    attention_levels=[False, True, True],
    norm_num_groups=32,
    norm_eps=1e-6,
    resblock_updown=True,
    num_head_channels=[0, 512, 768],
    with_conditioning=True,
    transformer_num_layers=1,
    cross_attention_dim=4,
    upcast_attention= True,
    use_flash_attention = False
)
diffusion.load_state_dict(torch.load("./pretrained_models/diffusion_model.pth"))
diffusion.eval()

scheduler = DDIMScheduler(beta_start=0.0015, beta_end=0.0205, num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)
scheduler.set_timesteps(num_inference_steps=50)

device = torch.device("cuda")
diffusion = diffusion.to(device)
aekl = aekl.to(device)


@torch.no_grad()
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
    latent_shape = torch.randn((1, 3, 20, 28, 20)).to(device)

    progress_bar = iter(scheduler.timesteps)
    image = latent_shape
    conditioning = cond.to("cuda").unsqueeze(1)
    cond_concat = conditioning.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(image.shape[2:]))
    for t in progress_bar:
        with autocast():
            model_output = diffusion(
                torch.cat((image, cond_concat), dim=1),
                timesteps=torch.Tensor((t,)).to(image.device).long(),
                context=conditioning,
            )
            image, _ = scheduler.step(model_output, t, image)

    with autocast():
        x_hat = aekl.decode_stage_2_outputs(image)

    return x_hat.cpu().numpy()


def create_videos_and_file(
    gender_radio,
    age_slider,
    ventricular_slider,
    brain_slider,
):
    output_dir = Path(f"/media/walter/Storage/Projects/gradio_medical_ldm/outputs/{str(uuid.uuid4())}")
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
    with mediapy.VideoWriter(f"{str(output_dir)}/brain_axial.mp4", shape=(150, 214), fps=12, crf=18) as w:
        for idx in range(image_data.shape[2]):
            img = image_data[:, :, idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(f"{str(output_dir)}/brain_sagittal.mp4", shape=(145, 214), fps=12, crf=18) as w:
        for idx in range(image_data.shape[0]):
            img = np.rot90(image_data[idx, :, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(f"{str(output_dir)}/brain_coronal.mp4", shape=(145, 150), fps=12, crf=18) as w:
        for idx in range(image_data.shape[1]):
            img = np.rot90(np.flip(image_data, axis=1)[:, idx, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    # Create file
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 96.48149872],
            [0.0, 1.0, 0.0, -141.47715759],
            [0.0, 0.0, 1.0, -156.55375671],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    empty_header = nib.Nifti1Header()
    sample_nii = nib.Nifti1Image(image_data, affine, empty_header)
    nib.save(sample_nii, f"{str(output_dir)}/my_brain.nii.gz")

    # time.sleep(2)

    return (
        f"{str(output_dir)}/brain_axial.mp4",
        f"{str(output_dir)}/brain_sagittal.mp4",
        f"{str(output_dir)}/brain_coronal.mp4",
        f"{str(output_dir)}/my_brain.nii.gz",
    )


# TEXT
title = "Brain imaging generation with latent diffusion models"
description = ""
article = """
<center><img src="https://raw.githubusercontent.com/Warvito/public_images/master/assets/Footer_1.png" alt="Project by amigos.ai" style="width:450px;"></center>
"""
# background: rgb(2 163 163);
demo = gr.Blocks(
    css="""
    #primary-button {border: rgb(2 163 163);background: rgb(2 163 163);background-color: rgb(2 163 163); color: white;},
    
    input[type='radio']:after {
        width: 15px;
        height: 15px;
        border-radius: 15px;
        top: -2px;
        left: -1px;
        position: relative;
        background-color: #d1d3d1;
        content: '';
        display: inline-block;
        visibility: visible;
        border: 2px solid white;
        background-color: rgb(2 163 163);
    }

    input[type='radio']:checked:after {
        width: 15px;
        height: 15px;
        border-radius: 15px;
        top: -2px;
        left: -1px;
        position: relative;
        background-color: #ffa500;
        content: '';
        display: inline-block;
        visibility: visible;
        border: 2px solid white;
      background-color: rgb(2 163 163);
    }
    
    input[type="range"] {
      -webkit-appearance: none;
      background-color: rgb(2 163 163);
    }
    
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      background-color: rgb(2 163 163);
    }
    
    input[type=range]::-webkit-slider-runnable-track  {
      -webkit-appearance: none;
      background-color: rgb(2 163 163);
    }

    """
)

with demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem; color: rgb(55 65 81); '><b>" + title + "</b></h1>")
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
                                label="Ventricular volume",
                                interactive=True,
                            )
                            brain_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                label="Brain volume",
                                interactive=True,
                            )
                        with gr.Row():
                            submit_btn = gr.Button("Generate", elem_id="primary-button")

        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Axial View"):
                        axial_sample_plot = gr.Video(show_label=False, format="mp4")
                    with gr.TabItem("Sagittal View"):
                        sagittal_sample_plot = gr.Video(show_label=False, format="mp4")
                    with gr.TabItem("Coronal View"):
                        coronal_sample_plot = gr.Video(show_label=False, format="mp4")

    gr.Markdown(article)

    submit_btn.click(
        create_videos_and_file,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )


demo.queue()
demo.launch()
