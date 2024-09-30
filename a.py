import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import streamlit as st
from skimage.filters import gabor
import os
import rarfile  # For extracting .rar files
import tempfile

# Streamlit app
st.title("NIfTI MRI Image Processor with Gabor Filter")
st.write("Upload a RAR archive (.rar) containing NIfTI (.nii) MRI image, apply the Gabor filter, and visualize 2D slices.")

# Function to extract RAR file
def extract_rar_file(rar_file_path, extract_dir):
    """Extracts the contents of a .rar file into a specified directory."""
    with rarfile.RarFile(rar_file_path) as rf:
        rf.extractall(path=extract_dir)
    st.write(f"Extracted files to: {extract_dir}")
    return extract_dir

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

# Upload RAR file
uploaded_rar_file = st.file_uploader("Upload a RAR file (.rar) containing NIfTI files", type=["rar"])

if uploaded_rar_file is not None:
    # Create a temporary directory to extract the RAR file
    temp_dir = tempfile.TemporaryDirectory()
    extract_dir = extract_rar_file(uploaded_rar_file, temp_dir.name)

    # Find NIfTI files in the extracted directory
    nii_files = [f for f in os.listdir(extract_dir) if f.endswith('.nii')]
    st.write(f"Found NIfTI files: {nii_files}")

    if len(nii_files) > 0:
        # Select NIfTI file to process
        selected_nii_file = st.selectbox("Select a NIfTI file to process", nii_files)
        nii_path = os.path.join(extract_dir, selected_nii_file)

        # Load the NIfTI file
        im_nii = nib.load(nii_path)
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
        gabor_output_path = f"gabor_output_{selected_nii_file}"
        slice_paths = save_slices_as_png(gabor_mask, f"gabor_{selected_nii_file}", output_dir)
        
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
