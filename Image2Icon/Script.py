import os
import json
from PIL import Image

# === Settings ===
source_dir = r"C:\Users\Javie\Documents\GitHub\Tools-that-we-all-need\Image2Icon\Images"
icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
output_base = os.path.join(source_dir, "GeneratedIcons")
os.makedirs(output_base, exist_ok=True)
record_file = os.path.join(source_dir, "processed_images.json")

# === Load previously processed files ===
if os.path.exists(record_file):
    with open(record_file, "r") as f:
        processed_images = json.load(f)
else:
    processed_images = {}

# === Process images ===
for filename in os.listdir(source_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")) and filename not in processed_images:
        try:
            image_path = os.path.join(source_dir, filename)
            img = Image.open(image_path)
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_base, base_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save one .ico with all sizes
            ico_all_path = os.path.join(output_dir, f"{base_name}_all.ico")
            img.save(ico_all_path, format='ICO', sizes=icon_sizes)

            # Save individual .ico files for each size
            for size in icon_sizes:
                resized_img = img.resize(size, Image.LANCZOS)
                ico_individual_path = os.path.join(output_dir, f"{base_name}_{size[0]}x{size[1]}.ico")
                resized_img.save(ico_individual_path, format='ICO', sizes=[size])

            # Update record
            processed_images[filename] = True
            print(f"✅ Processed: {filename}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# === Save record ===
with open(record_file, "w") as f:
    json.dump(processed_images, f, indent=4)

print("🎉 All done!")
