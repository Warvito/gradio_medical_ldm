import nibabel as nib

model = None

def sample():
    image = nib.load("/home/walter/Downloads/sub-HCD0001305_ses-01_space-MNI152NLin2009aSym_T2w.nii.gz")
    image_data = image.get_fdata()
    return image_data