import numpy as np
import plotly.graph_objects as go
import nibabel as nib


def draw(a,b,c,d):
    selected_plane = 2
    image = nib.load("/media/walter/Storage/Downloads/sub-000000_T1w.nii.gz")
    image_data = image.get_fdata()
    image_data = image_data[5:-5,5:-5,:-15]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    volume = image_data
    volume = np.swapaxes(volume,selected_plane,0)
    r, c = volume[0].shape

    nb_frames = volume.shape[0]
    z_max = (nb_frames-1)/10.0

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(z_max - k * 0.1) * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[(nb_frames-1) - k]),
                    cmin=0,
                    cmax=1,
                    showscale=False
                ),
                name=str(k)  # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ]
    )

    fig.add_trace(
        go.Surface(
            z=z_max * np.ones((r, c)),
            surfacecolor=np.flipud(volume[(nb_frames-1)]),
            colorscale='Gray',
            cmin=0,
            cmax=1,
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
            "pad": {"b": 10, "t": 10},
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
        width=900,
        height=900,
        scene=dict(
            zaxis=dict(
                range=[-0.1, (nb_frames)/10.0],
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
                "pad": {"r": 10, "t": 20},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    return fig