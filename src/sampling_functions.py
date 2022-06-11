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