import gradio as gr


def greet(name):
    return "Hello !!"

demo = gr.Interface(
    fn=greet,
    inputs=[
gr.Radio(
        choices=["Female", "Male"],
        value="Female",
        type="index",
        label="Gender",
        interactive=True,
    ),
        gr.Slider(0, 100),
        gr.Slider(0, 100),
        gr.Slider(0, 100)
    ],
    outputs="text"
)
demo.launch(share=False)
