# Squad Data Analysis

This project analyzes SQuAD dataset statistics and generates visualizations.

## Setup

1. Create a virtual environment (already set up):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Download the spaCy English model:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the main script:
```powershell
python main.py
```

## Project Structure
- `main.py`: Main analysis script
- `data.json`: Input data
- `plots/`: Output plots
- `*.md`: Documentation and stats

