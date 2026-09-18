import streamlit as st
import torch
import numpy as np
from PIL import Image

from model import SiameseUNet


# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="UrbanLens",
    page_icon="🛰️",
    layout="wide"
)


# -------------------------
# Load Model
# -------------------------

device = torch.device("cpu")

model = SiameseUNet().to(device)

model.load_state_dict(
    torch.load(
        "urbanlens_siamese_unet.pth",
        map_location=device
    )
)

model.eval()


# -------------------------
# Title
# -------------------------

st.title("🛰️ UrbanLens")
st.subheader(
    "AI-Powered Bitemporal Satellite Change Detection"
)

st.write(
    "Upload two satellite images of the same area "
    "captured at different time periods to detect changes."
)


# -------------------------
# Image Upload
# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Time 1")
    image_A = st.file_uploader(
        "Upload the earlier satellite image",
        type=["png", "jpg", "jpeg"],
        key="time1"
    )

with col2:
    st.markdown("### 📅 Time 2")
    image_B = st.file_uploader(
        "Upload the later satellite image",
        type=["png", "jpg", "jpeg"],
        key="time2"
    )


# -------------------------
# Prediction
# -------------------------

if image_A is not None and image_B is not None:

    time1 = Image.open(image_A).convert("RGB")
    time2 = Image.open(image_B).convert("RGB")

    st.divider()

    # Display uploaded images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Time 1")
        st.image(time1, use_container_width=True)

    with col2:
        st.markdown("### Time 2")
        st.image(time2, use_container_width=True)

    # Predict button
    if st.button(
        "🔍 Detect Changes",
        use_container_width=True
    ):

        # Resize exactly like training
        input_A = time1.resize(
            (256, 256),
            Image.Resampling.BILINEAR
        )

        input_B = time2.resize(
            (256, 256),
            Image.Resampling.BILINEAR
        )

        # Convert to NumPy
        input_A = np.array(input_A)
        input_B = np.array(input_B)

        # Convert to tensor
        input_A = torch.from_numpy(
            input_A
        ).float().permute(2, 0, 1) / 255.0

        input_B = torch.from_numpy(
            input_B
        ).float().permute(2, 0, 1) / 255.0

        # Add batch dimension
        input_A = input_A.unsqueeze(0).to(device)
        input_B = input_B.unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():

            prediction = model(
                input_A,
                input_B
            )

            probability = torch.sigmoid(
                prediction
            )

        # Threshold
        predicted_mask = (
            probability > 0.5
        ).float()

        # Remove batch/channel dimensions
        mask = predicted_mask[
            0, 0
        ].cpu().numpy()

        # Calculate percentage
        changed_pixels = np.sum(mask)
        total_pixels = mask.size

        change_percentage = (
            changed_pixels / total_pixels
        ) * 100

        # -------------------------
        # Create Overlay
        # -------------------------

        time2_array = np.array(
            input_B[0]
            .cpu()
            .permute(1, 2, 0)
        )

        overlay = time2_array.copy()

        alpha = 0.45

        red_layer = np.zeros_like(
            overlay
        )

        red_layer[:, :, 0] = 1.0

        changed = mask == 1

        overlay[changed] = (
            (1 - alpha)
            * overlay[changed]
            + alpha
            * red_layer[changed]
        )

        # -------------------------
        # Results
        # -------------------------

        st.divider()

        st.header("🔍 Change Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Predicted Change")
            st.image(
                mask,
                clamp=True,
                use_container_width=True
            )

        with col2:
            st.markdown("### Change Overlay")
            st.image(
                overlay,
                clamp=True,
                use_container_width=True
            )

        with col3:
            st.markdown("### Change Summary")
            st.metric(
                "Approx. Changed Area",
                f"{change_percentage:.2f}%"
            )

            st.write(
                "The highlighted regions represent "
                "areas predicted as changed between "
                "the two satellite images."
            )