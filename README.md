# German to English Category Translator

This Python script runs completely offline after installing once it translates German categories and their descriptions to English using the MarianMT model from Hugging Face's transformers library.

## Features

- Translates both category labels and descriptions from German to English
- Processes data in batches for efficient memory usage
- Saves progress automatically
- Optimized for M2 processors
- Supports offline model usage

## Requirements

- Python 3.x
- pandas
- transformers
- torch
- tqdm
- psutil

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input CSV file named 'data.csv' in the project directory
2. The CSV should have at least two columns: 'label' and 'text'
3. Run the script:
```bash
python translate_categories.py
```

The script will create a new Excel file 'translated_categories.xlsx' with the English translations.

## Notes

- The script uses the Helsinki-NLP/opus-mt-de-en model for translation
- Progress is saved every 1000 rows
- Temporary files are automatically cleaned up 
