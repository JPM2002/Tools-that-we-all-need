import os
import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image

# Define the URL or data for the QR code
data = "https://www.linkedin.com/in/javier-pozo-miranda/"

# Create a "media" folder in the current working directory if it doesn't exist
media_folder = os.path.join(os.getcwd(), "media")
os.makedirs(media_folder, exist_ok=True)

# Create a QRCode object with high error correction
qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR code (higher number = larger code)
    error_correction=ERROR_CORRECT_H,  # High error correction allows embedding images
    box_size=12,  # Size of each box in pixels
    border=4,  # Thickness of the border (minimum is 4)
)

# Add data to the QR code
qr.add_data(data)
qr.make(fit=True)

# Generate the QR code image with custom colors
qr_img = qr.make_image(fill_color="white", back_color="black")

# Embed a logo in the center of the QR code
logo_path = "logo.png"  # Replace with the path to your logo image
try:
    logo = Image.open(logo_path)

    # Resize the logo to fit inside the QR code
    logo_size = (qr_img.size[0] // 4, qr_img.size[1] // 4)
    logo = logo.resize(logo_size, Image.ANTIALIAS)

    # Calculate the position for the logo to be centered
    logo_position = (
        (qr_img.size[0] - logo_size[0]) // 2,
        (qr_img.size[1] - logo_size[1]) // 2,
    )

    # Paste the logo onto the QR code
    qr_img.paste(logo, logo_position, mask=logo)
except FileNotFoundError:
    print("Logo file not found. Generating QR code without logo.")

# Save the QR code with the logo in the "media" folder
file_path = os.path.join(media_folder, "custom_qrcode.png")
qr_img.save(file_path)

print(f"Custom QR code saved to: {file_path}")
