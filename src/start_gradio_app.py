import gradio as gr
import mlflow.pytorch
import plotly.express as px
import torch

from models.ddim import DDIMSampler

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

    return x_hat


# TEXT
title = "Amigo's Brains with Diffusion Models"
description = """
Test our diffusion models trained to generate brain imaging data! 
<a href="https://arxiv.org/abs/2112.00726"><img style="display:inline" src="https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg"></a>
"""

article = """
Checkout our dataset with [100K synthetic brain](https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b)! 🧠🧠🧠

By [Walter Hugo Lopez Pinaya](https://twitter.com/warvito) from [AMIGO](https://amigos.ai/)
<center><a href="https://amigos.ai/"><img src="https://amigos.ai/assets/images/logo_dark_rect.png" width=300px></a></center>
"""


def get_fig(a, b, c, d):
    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
                 animation_frame="year", animation_group="country", range_y=[0, 4000000000])

    fig2 = px.bar(df, x="continent", y="pop", color="continent",
                 animation_frame="year", animation_group="country", range_y=[0, 4000000000])
    fig3 = px.bar(df, x="continent", y="pop", color="continent",
                 animation_frame="year", animation_group="country", range_y=[0, 4000000000])
    return fig, fig2, fig3


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
                            unrest_gender_radio = gr.Number(
                                value=1.0,
                                precision=1,
                                label="Gender [Female=0, Male=1]",
                                interactive=True,
                            )
                            unrest_age_slider = gr.Number(
                                value=63,
                                precision=1,
                                label="Age [years]",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_ventricular_slider = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of ventricular cerebrospinal fluid",
                                interactive=True,
                            )
                            unrest_brain_slider = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of brain",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Axial View"):
                        axial_sample_plot = gr.Plot()
                    with gr.TabItem("Sagittal View"):
                        sagittal_sample_plot = gr.Plot()
                    with gr.TabItem("Coronal View"):
                        coronal_sample_plot = gr.Plot()
    gr.Markdown(article)

    submit_btn.click(
        get_fig,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )
    unrest_submit_btn.click(
        get_fig,
        [
            unrest_gender_radio,
            unrest_age_slider,
            unrest_ventricular_slider,
            unrest_brain_slider,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )
    # examples = gr.Examples(
    #     examples=[
    #         [1.0, 63, 0.9, 0.5],
    #         [0.0, 63, 0.1, 0.5],
    #         [1.0, 45, 0.5, 0.5],
    #     ],
    #     inputs=[
    #         gender_radio,
    #         age_slider,
    #         ventricular_slider,
    #         brain_slider,
    #     ]
    # )

demo.launch()
