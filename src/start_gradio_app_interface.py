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
"""

article = """
Checkout our dataset with [100K synthetic brain](https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b)!

By Walter Hugo Lopez Pinaya from [AMIGO](https://amigos.ai/)
<center><img src="https://amigos.ai/assets/images/logo_dark_rect.png" width=300px></center>
"""


def get_fig(a, b, c, d):
    df = px.data.gapminder()

    fig = px.bar(df, x="continent", y="pop", color="continent",
                 animation_frame="year", animation_group="country", range_y=[0, 4000000000])
    return fig


demo = gr.Interface(
    fn=get_fig,
    inputs=[
        gr.Radio(
            choices=["Female", "Male"],
            value="Female",
            type="index",
            label="Gender",
            interactive=True,
        ),
        gr.Slider(
            minimum=44,
            maximum=82,
            value=63,
            label="Age [years]",
            interactive=True,
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            label="Volume of ventricular cerebrospinal fluid",
            interactive=True,
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            label="Volume of brain, grey+white matter (normalised for head size)",
            interactive=True,
        )
    ],
    outputs=gr.Plot(),
    examples=[
        [1.0, 63, 0.9, 0.5],
        [0.0, 63, 0.1, 0.5],
        [1.0, 45, 0.5, 0.5],
    ],
    title=title,
    description=description,
    article=article,
    allow_flagging=False
)

demo.launch()
