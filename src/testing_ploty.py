import numpy as np
import plotly.graph_objects as go

from skimage import io

vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
volume = vol.T
r, c = volume[0].shape

nb_frames = 68

fig = go.Figure(
    frames=[
        go.Frame(
            data=go.Surface(
                z=(6.7 - k * 0.1) * np.ones((r, c)),
                surfacecolor=np.flipud(volume[67 - k]),
                cmin=0,
                cmax=200,
                showscale=False
            ),
            name=str(k)  # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)
    ]
)

fig.add_trace(
    go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[67]),
        colorscale='Gray',
        cmin=0,
        cmax=200,
        showscale=False
    )
)


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


sliders = [
    {
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f.name], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig.frames)
        ],
    }
]

# Layout
fig.update_layout(
    showlegend=False,
    width=1200,
    height=1200,
    scene=dict(
        zaxis=dict(
            range=[-0.1, 6.8],
            autorange=False,
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        aspectratio=dict(x=2, y=2, z=1),
        camera=dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=1.0),
            projection=dict(type="orthographic"),
        ),
        dragmode=False,
        hovermode=False,
    ),
    updatemenus=[
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
            "x": 0.1,
            "y": 0,
        }
    ],
    sliders=sliders
)

config = {
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'orbitRotation',
        'tableRotation',
        "resetCameraDefault3d",
        "zoom3d",
        "pan3d",
        "toImage"
    ],
    'displayModeBar': True,
}

fig.show(config=config)
