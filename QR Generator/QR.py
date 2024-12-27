import os
import qrcode

# The URL for the QR code
url = "https://www.linkedin.com/in/javier-pozo-miranda/"

# Create a folder called "media" in the current working directory if it doesn't exist
media_folder = os.path.join(os.getcwd(), "media")
os.makedirs(media_folder, exist_ok=True)

# Generate the QR code
img = qrcode.make(url)

# Save the QR code in the "media" folder
file_path = os.path.join(media_folder, "qrcode.png")
img.save(file_path)

print(f"QR code saved to: {file_path}")
