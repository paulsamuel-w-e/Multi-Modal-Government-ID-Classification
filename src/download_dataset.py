import gdown
import zipfile
import os

# Google Drive file ID (Extracted from your link)
file_id = "1Gu23xr357BPzGoocyPw6IPUhnz5mf52j"

# Destination file name
output = "file.zip"

# Download the zip file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Extract the zip file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("Data")

# Remove the zip file after extraction
os.remove(output)