import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import gabor
import os

# Streamlit app
st.title("NIfTI MRI Image Processor with Gabor Filter")
st.write("Upload a NIfTI (.nii) MRI image, apply the Gabor filter, and visualize 2D slices.")

# Function to save 2D slices as PNG
def save_slices_as_png(volume, filename_prefix, output_dir, axis=2):
    """Saves slices of a 3D volume as PNG images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_slices = volume.shape[axis]
    slice_paths = []
    for i in range(num_slices):
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:
            slice_2d = volume[:, :, i]

        plt.imshow(slice_2d, cmap='gray')
        plt.axis('off')
        slice_filename = f"{filename_prefix}_slice_{i:03d}.png"
        slice_filepath = os.path.join(output_dir, slice_filename)
        plt.savefig(slice_filepath, bbox_inches='tight', pad_inches=0)
        plt.close()
        slice_paths.append(slice_filepath)
    return slice_paths

# Upload NIfTI file
uploaded_file = st.file_uploader("Upload a NIfTI file (.nii)", type=["nii"])

if uploaded_file is not None:
    # Load the NIfTI file
    im_nii = nib.load(uploaded_file)
    im = im_nii.get_fdata().squeeze()
    st.write(f"Loaded image with shape: {im.shape}")

    # Gabor filter parameters
    gabor_frequency = st.slider("Gabor Filter Frequency", min_value=0.05, max_value=0.5, value=0.1, step=0.01)
    gabor_threshold = st.slider("Gabor Filter Threshold", min_value=0.0, max_value=0.05, value=0.001, step=0.001)

    # Apply Gabor filter
    st.write("Applying Gabor filter...")
    gabor_filtered_volume = np.zeros_like(im)
    for slice_idx in range(im.shape[2]):  # Iterate over each slice in the 3D volume
        real, _ = gabor(im[:, :, slice_idx], frequency=gabor_frequency)
        gabor_filtered_volume[:, :, slice_idx] = real

    # Apply threshold
    gabor_mask = gabor_filtered_volume > gabor_threshold
    gabor_mask = gabor_mask.astype(np.float32)

    # Visualize slices
    st.write("Visualize Gabor-filtered slices:")
    selected_slice = st.slider("Select slice number", min_value=0, max_value=im.shape[2] - 1, value=0)

    # Original slice
    st.write("Original slice:")
    st.image(im[:, :, selected_slice], clamp=True, channels="GRAY")

    # Gabor filtered slice
    st.write("Gabor-filtered slice:")
    st.image(gabor_mask[:, :, selected_slice], clamp=True, channels="GRAY")

    # Save output images
    output_dir = "gabor_output_slices"
    gabor_output_path = f"gabor_output_{uploaded_file.name}"
    slice_paths = save_slices_as_png(gabor_mask, f"gabor_{uploaded_file.name}", output_dir)
    
    st.write(f"Gabor-filtered volume saved in: {gabor_output_path}")
    st.write(f"2D slice images saved in: {output_dir}")

    # Allow downloading processed slices
    with open(slice_paths[selected_slice], "rb") as file:
        st.download_button(label="Download selected slice as PNG", data=file, file_name=f"gabor_slice_{selected_slice:03d}.png", mime="image/png")

    # Save NIfTI file
    nib.save(nib.Nifti1Image(gabor_mask, affine=im_nii.affine), gabor_output_path)
    with open(gabor_output_path, "rb") as file:
        st.download_button(label="Download Gabor-filtered volume as NIfTI", data=file, file_name=gabor_output_path, mime="application/octet-stream")

    st.success("Processing complete. You can download the filtered volume and individual slices.")
