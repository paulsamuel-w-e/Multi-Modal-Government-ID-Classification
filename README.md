# Document-Classification
-------------------------
# Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

-------------------------
## System Requirements

- Operating System: Windows 11 (tested)

- GPU: NVIDIA RTX 3050 (for GPU-accelerated OCR tasks)

- CUDA Toolkit: CUDA 12.6 or 12.8 (verified with PaddleOCR)

- Python: Version 3.11.5 (64-bit)

- Processor Architecture: x86_64 / x64 / Intel 64 / AMD64

- Python & pip: Must both be 64-bit
 
## use this code to download the dataset into your codespace
```
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
```
or download the dataset from [Google Drive](https://drive.google.com/file/d/1Gu23xr357BPzGoocyPw6IPUhnz5mf52j/view?usp=sharing)
----------------------------
## Optical Character Recognition
This project utilizes PaddleOCF by Alibaba Cloud
Three Major New Features in PaddleOCR 3.0:

- 🖼️ Universal-Scene Text Recognition Model: A single model that handles five different text types plus complex handwriting. Overall recognition accuracy has increased by 13 percentage points over the previous generation.

- 🧮 General Document-Parsing Solution: Delivers high-precision parsing of multi-layout, multi-scene PDFs, outperforming many open- and closed-source solutions on public benchmarks.

- 📈 Intelligent Document-Understanding Solution: Natively powered by the ERNIE 4.5 Turbo, achieving 15 percentage points higher accuracy than its predecessor
## Prerequisites
### Environmental Preparation
Create a separate environment for paddleocr
- here in this project we use (python 3.11.5)
```bash
python -m venv env/OCRenv
```
Activate environment:
```
env\OCRenv\Scripts\activate.bat
```

### 1.1 How to Check Your Environment

#### Confirm Python Version

Ensure you are using one of the following supported versions:

- Python 3.8 / 3.9 / 3.10 / 3.11 / 3.12 / 3.13
- here in this project we use (python 3.11.5)

Check your version with:

```bash
python --version
```

---

#### Confirm pip Version

Ensure pip is version 20.2.2 or above:

```bash
python -m pip --version
```

---

#### Confirm Python & pip Architecture

Python and pip must be 64-bit, and the processor architecture should be `x86_64`, `x64`, `Intel 64`, or `AMD64`.

Run this command to verify:

```bash
python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```

Expected Output:

- First line: `64bit`
- Second line: `x86_64`, `x64`, or `AMD64`

---

### Platform Limitations

- NCCL and distributed training are not supported on Windows.

---

### Hardware Compatibility

- The default installation package requires MKL (Intel’s Math Kernel Library).
- All Intel chips support MKL.

###To install the model into your device
- For Thorough Installation for your system requirements check on [PaddlePaddle official installation documentation](https://www.paddlepaddle.org.cn/install/quick).

1. we installed GPU Version of PaddlePaddle (here we use CUDA 12.6)

```bash
 python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
For other systems or cpu version, refer to [PaddlePaddle official installation documentation](https://www.paddlepaddle.org.cn/install/quick).

2. install `paddleocr`
```bash
python -m pip install paddleocr
```

3. install `chocolatey` and `ccache` (optional)
open powershell with admin previlege and copy paste this command into your shell and enter [for assistance check official installation documentation](https://docs.chocolatey.org/en-us/choco/setup/#more-install-options)
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
 Upgrade Chocolatey
```
choco upgrade chocolatey
```

 Install ccache
```
choco install ccache

```
 Check Installation
```
where ccache
```
 If ccache isn't found after install, you may need to manually add chocolatey to your PATH
 you can find it in `C:\ProgramData\chocolatey\bin`
