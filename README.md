cd# Modular & Agentic RAG Document Assistant

A fully local, offline Retrieval-Augmented Generation (RAG) document assistant built for multi-PDF technical and policy analysis. The application combines hierarchical separator chunking, a three-stage hybrid retrieval funnel (**FAISS MMR + BM25 + Cross-Encoder Reranking**), and an autonomous **Llama 3 Agentic Layer** (Relevance Gatekeeper & Query Rewriter) inside a local **Streamlit** interface.

## 📁 Project Contents

Once extracted, the project directory contains the following files and folders:

```text
<project-folder>/
├── app.py                                      # Main local Streamlit application (Modular & Agentic RAG)
├── requirements.txt                            # Python package dependencies
├── README.md                                   # Setup and execution instructions
├── *.ipynb                                     # Jupyter Notebooks (Naive RAG Baseline & Modular/Agentic RAG benchmarks)
└── Test Results/                               # Benchmark CSV scorecards, comparative charts, and Word report (.docx)
```

*(Note: The `faiss_modular_index/` folder is intentionally excluded from the archive to keep the submission lightweight; it will be generated automatically inside this directory the first time you run `app.py`.)*

## 📋 Prerequisites

Before running the application, please ensure your machine has:

* **Python 3.10 or higher** installed.
* **Ollama** installed for local LLM inference.
* **Internet access for initial setup only** (to install Python packages and download the local Ollama and HuggingFace models). Once downloaded, the application runs **100% locally and offline**.

## 🚀 Step-by-Step Setup & Execution Guide (Windows, macOS & Linux)

### Step 1: Extract the Project Archive

1. Download and extract the `.zip` archive to a convenient location on your computer (for example, your Desktop or Documents folder).
2. Open the extracted folder and verify that `app.py`, `requirements.txt`, `README.md`, `PDF Test Files/`, the Jupyter notebooks, and the test results are all located directly inside it.

### Step 2: Install Ollama & Pull Llama 3

This chatbot uses **Llama 3** running locally via Ollama—no API keys or cloud services are required.

#### 🪟 Windows

1. Download and run the Windows installer from [**https://ollama.com/download**](https://ollama.com/download).
2. Confirm the Ollama icon is running in your system tray (bottom-right taskbar).
3. Open **Command Prompt** or **PowerShell** and download Llama 3:

```powershell
ollama pull llama3
```

#### 🍎 macOS

1. Download the macOS app from [**https://ollama.com/download**](https://ollama.com/download), unzip it, and drag **Ollama** into your **Applications** folder (or install via Homebrew with `brew install ollama`).
2. Launch **Ollama** from Applications so it runs in the top menu bar.
3. Open **Terminal** and download Llama 3:

```bash
ollama pull llama3
```

#### 🐧 Linux

1. Open a terminal and install Ollama using the official install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. Ensure the Ollama background service is running (if it does not start automatically, run `ollama serve` in a separate terminal tab or `sudo systemctl start ollama`), then pull Llama 3:

```bash
ollama pull llama3
```

### Step 3: Navigate to the Application Directory

Open your terminal (or the VS Code integrated terminal) and change directories (`cd`) into the extracted project folder.

* **🪟 Windows (Command Prompt / PowerShell):**

```powershell
cd "C:\Users\YourUsername\Desktop\RAG_Capstone_Project"
dir
```

* **🍎 macOS (Terminal):**

```bash
cd ~/Desktop/RAG_Capstone_Project
ls
```

* **🐧 Linux (Terminal):**

```bash
cd ~/Desktop/RAG_Capstone_Project
ls
```

*(Verify that `app.py` and `requirements.txt` appear in the directory listing before continuing.)*

### Step 4: Create & Activate a Virtual Environment (Recommended)

Creating a virtual environment isolates the project's dependencies from your system Python installation. Run the commands for your operating system from inside the project folder:

* **🪟 Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell displays a script execution policy error, either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or switch to **Command Prompt** and use `.\.venv\Scripts\activate.bat` instead. If `python` is not recognized, substitute `py`.)*

* **🪟 Windows (Command Prompt):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

* **🍎 macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

* **🐧 Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(On Debian/Ubuntu, if `python3 -m venv` reports a missing `venv` package, install it first with `sudo apt update && sudo apt install -y python3-venv`.)*

When active, your terminal prompt will show `(.venv)` at the beginning of the line.

### Step 5: Install Python Dependencies

Using `python -m pip` guarantees that all packages install into the exact same Python interpreter you will use to launch Streamlit.

* **🪟 Windows:**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*(If `python` is not recognized outside a virtual environment, use `py -m pip install -r requirements.txt`.)*

* **🍎 macOS & 🐧 Linux:**

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Step 6: Launch the Local Application

Ensure **Ollama is running in the background**, then launch the local Streamlit interface from inside the project directory:

* **🪟 Windows:**

```powershell
python -m streamlit run app.py
```

*(Or strictly local with telemetry disabled:)*

```powershell
python -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

* **🍎 macOS & 🐧 Linux:**

```bash
python3 -m streamlit run app.py
```

*(Or strictly local with telemetry disabled:)*

```bash
python3 -m streamlit run app.py --server.address localhost --browser.gatherUsageStats false
```

Your default web browser will automatically open the local interface at **`http://localhost:8501`**.

> **Note on First Launch:** On the very first startup, the application will automatically download two lightweight local HuggingFace models (`all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`) and automatically index the PDFs in `./PDF Test Files` to generate the `faiss_modular_index/` directory (~60 seconds). All subsequent launches will load the saved index almost instantaneously.

## 🖥️ Using the Application

* **Ask Questions:** Type any question related to the uploaded PDFs in the chat input bar at the bottom.
* **Live Agentic Telemetry:** Expand the status box during or after a query to inspect Pass 1 retrieval, the **Relevance Gatekeeper** verdict (`YES`/`NO`), any self-correcting **Pass 2 Query Rewrites**, and exact retrieval vs. generation latencies.
* **Inspect Source Evidence:** Click **"Inspect Retrieved Source Chunks"** beneath any answer to verify the exact PDF filenames, page numbers, and raw excerpts used.
* **Sidebar Controls:**
  * **Reranker Top-N Chunks Slider:** Adjust how many reranked chunks are passed to Llama 3 (default: `6`).
  * **Agentic Gate Toggle:** Enable or disable the pre-generation Relevance Gatekeeper and Query Rewriter.
  * **PDF Corpus Manager:** Drag and drop additional `.pdf` files to incrementally index them into the live knowledge base.

## 🛠️ Troubleshooting

| Issue | Likely Cause | Fix |
| :--- | :--- | :--- |
| `No module named streamlit` | `pip` installed packages into a different Python environment than the active terminal. | Activate `.venv` and run both install and launch with the same interpreter (`python -m ...` on Windows, `python3 -m ...` on macOS/Linux). |
| PowerShell blocks `Activate.ps1` (Windows) | Windows default script execution policy. | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or use Command Prompt (`.\.venv\Scripts\activate.bat`). |
| `python3 -m venv` fails (Linux) | System `venv` module is not installed on Debian/Ubuntu. | Run `sudo apt update && sudo apt install -y python3-venv` and retry Step 4. |
| Connection error / Ollama refused | Ollama is not running or `llama3` was not pulled. | Start Ollama (app on Windows/macOS, or `ollama serve` on Linux) and confirm `ollama list` shows `llama3`. |
| Slow first startup | First-time download of HuggingFace embedding/reranker weights and automatic creation of `faiss_modular_index/`. | Wait ~1–2 minutes for the first initialization; models and the FAISS index are cached locally afterward. |
| Browser does not open automatically | Local security or default browser settings (common on headless/minimal Linux). | Manually open any browser and navigate to `http://localhost:8501`. |