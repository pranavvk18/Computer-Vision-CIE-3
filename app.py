import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 1. U-NET MODEL
# =====================================================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), CBR(128, 256), CBR(256, 256))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), CBR(256, 512), CBR(512, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))


# =====================================================
# 2. DeblurGAN-v2 Generator (ResNet-based)
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class DeblurGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residual=9):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_blocks(x)
        x = self.decoder(x)
        return x


# =====================================================
# 3. Model Paths Configuration
# =====================================================
MODEL_PATHS = {
    "U-Net": "D:/SEM-7/Computer Vision/CIE-3 CV/unet_deblur-2.pth",
    "DeblurGAN-v2": "D:/SEM-7/Computer Vision/CIE-3 CV/checkpoint_epoch_50.pth"
}


# =====================================================
# 4. Preprocessing & Postprocessing
# =====================================================
# For DeblurGAN-v2: normalize to [-1, 1] range
transform_deblurgan = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# For U-Net: normalize to [0, 1] range
transform_unet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def deblur_image(model, img, model_choice):
    model.eval()
    
    # Use appropriate transform based on model
    if model_choice == "DeblurGAN-v2":
        img_tensor = transform_deblurgan(img).unsqueeze(0).to(device)
    else:
        img_tensor = transform_unet(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze().cpu()
    
    # Denormalize based on model type
    if model_choice == "DeblurGAN-v2":
        # Denormalize from [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
    
    output = output.permute(1, 2, 0).numpy()
    output = np.clip(output * 255, 0, 255).astype("uint8")

    return Image.fromarray(output)


def load_model(model_choice):
    """Load the selected model with its weights"""
    model_path = MODEL_PATHS[model_choice]
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the model file is in the correct location.")
        return None
    
    try:
        if model_choice == "U-Net":
            model = UNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:  # DeblurGAN-v2
            model = DeblurGenerator().to(device)
            # Load checkpoint which contains generator_state_dict
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'generator_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['generator_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# =====================================================
# 5. Streamlit UI
# =====================================================
st.set_page_config(page_title="Image Deblurring App", page_icon="📸", layout="wide")

st.title("📸 Image Deblurring App")
st.write("Upload a blurred image and select the model you want to use for deblurring.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Select Model", ["U-Net", "DeblurGAN-v2"])
    
    st.divider()
    
    # Model information
    st.subheader("📊 Model Info")
    if model_choice == "U-Net":
        st.info("**U-Net**: Encoder-decoder architecture with skip connections. Good for general deblurring tasks.")
    else:
        st.info("**DeblurGAN-v2**: ResNet-based generator with residual blocks. State-of-the-art performance on motion blur.")
    
    st.divider()
    
    # Device info
    st.subheader("💻 Device Info")
    st.text(f"Using: {device.upper()}")
    if device == "cuda":
        st.success("✅ GPU acceleration enabled")
    else:
        st.warning("⚠️ Running on CPU (slower)")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    
    # Display model status
    model_path = MODEL_PATHS[model_choice]
    if os.path.exists(model_path):
        st.success(f"✅ Model loaded: `{os.path.basename(model_path)}`")
    else:
        st.error(f"❌ Model not found: `{model_path}`")
        st.info("Please update the MODEL_PATHS dictionary with correct paths.")
    
    # Upload blurred image
    image_file = st.file_uploader("Upload Blurred Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Original Blurred Image", use_column_width=True)
        
        # Image info
        st.caption(f"Size: {img.size[0]} x {img.size[1]} pixels")

with col2:
    st.subheader("📤 Output")
    
    if image_file:
        # Deblur button
        if st.button("🔄 Deblur Image", type="primary"):
            model = load_model(model_choice)
            
            if model is not None:
                with st.spinner(f"Processing with {model_choice}..."):
                    output = deblur_image(model, img, model_choice)
                
                st.image(output, caption="Deblurred Image", use_column_width=True)
                st.success("✅ Image deblurred successfully!")
                
                # Download button
                from io import BytesIO
                buf = BytesIO()
                output.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="💾 Download Deblurred Image",
                    data=byte_im,
                    file_name="deblurred_image.png",
                    mime="image/png"
                )
            else:
                st.error("Failed to load model. Please check the error messages.")
    else:
        st.info("👈 Upload an image to get started")

# Footer
st.divider()
st.caption("Built with Streamlit • Powered by PyTorch")