import os
from PyPDF2 import PdfMerger

# Path to the folder containing PDFs
folder_path = r"C:\Users\Javie\Documents\GitHub\Tools-that-we-all-need\Combine Pdfs\Chamaber"

# Output file name
output_file = os.path.join(folder_path, "merged_output.pdf")

# Initialize merger
merger = PdfMerger()

# Get all PDF files in the folder (sorted alphabetically)
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
pdf_files.sort()

# Add each PDF to the merger
for pdf in pdf_files:
    pdf_path = os.path.join(folder_path, pdf)
    merger.append(pdf_path)
    print(f"Added: {pdf}")

# Write out the merged PDF
merger.write(output_file)
merger.close()

print(f"\n✅ All PDFs merged into: {output_file}")
