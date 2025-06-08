import pandas as pd
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import gc
import os
import psutil
import warnings
import glob
warnings.filterwarnings('ignore')

def cleanup_temp_files():
    """Remove all temporary translation files"""
    temp_files = glob.glob('translated_categories_temp_*.xlsx')
    for file in temp_files:
        try:
            os.remove(file)
        except:
            pass

def get_available_memory():
    """Get available system memory in GB"""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)

def download_model_if_not_exists():
    """Download and save the model if it doesn't exist locally"""
    model_name = "Helsinki-NLP/opus-mt-de-en"
    local_model_path = "local_model"
    
    if not os.path.exists(local_model_path):
        print("Downloading model for offline use...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        # Quantize the model to reduce size and improve performance
        print("Optimizing model for M2 processor...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Save model and tokenizer locally
        print("Saving model locally...")
        model.save_pretrained(local_model_path)
        tokenizer.save_pretrained(local_model_path)
        print("Model saved successfully!")
    else:
        print("Using locally saved model...")
    
    return local_model_path

def translate_batch(texts, model, tokenizer, device, batch_size=32):
    """Translate a batch of texts for better performance"""
    try:
        # Tokenize the batch
        batch = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128  # Increased for M2's capabilities
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Generate translation with optimized parameters for M2
        with torch.no_grad():
            translated = model.generate(
                **batch,
                max_length=128,
                num_beams=2,  # Increased for better quality on M2
                early_stopping=True,
                do_sample=False,
                num_return_sequences=1,
                length_penalty=0.8  # Balanced for quality and speed
            )
        
        # Decode the translation
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Clear memory
        del batch, translated
        gc.collect()
        
        return translated_texts
    except Exception as e:
        print(f"Error in batch translation: {str(e)}")
        return texts  # Return original texts if translation fails

def process_batch(batch_df, model, tokenizer, device):
    """Process a batch of rows"""
    # Translate labels
    labels = batch_df['label'].tolist()
    translated_labels = translate_batch(labels, model, tokenizer, device)
    
    # Translate texts
    texts = batch_df['text'].tolist()
    translated_texts = translate_batch(texts, model, tokenizer, device)
    
    # Update the dataframe
    batch_df['label_english'] = translated_labels
    batch_df['text_english'] = translated_texts
    return batch_df

def main():
    # Clean up any existing temporary files
    cleanup_temp_files()
    
    # Set memory optimization for M2
    torch.set_num_threads(4)  # M2 can handle more threads
    
    # Load the CSV file
    print("Loading data...")
    df = pd.read_csv('data.csv')
    total_rows = len(df)
    print(f"Processing {total_rows} rows...")
    
    # Load the model from local storage
    print("Loading translation model...")
    local_model_path = download_model_if_not_exists()
    tokenizer = MarianTokenizer.from_pretrained(local_model_path)
    model = MarianMTModel.from_pretrained(local_model_path)
    
    # Move model to CPU and set to evaluation mode
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Create new columns for English translations
    df['label_english'] = ''
    df['text_english'] = ''
    
    try:
        # Process in larger batches for M2
        batch_size = 32  # Increased batch size for M2
        num_batches = (total_rows + batch_size - 1) // batch_size
        
        print("Starting translation...")
        for i in tqdm(range(0, total_rows, batch_size), total=num_batches, desc="Processing batches"):
            # Get batch
            end_idx = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:end_idx].copy()
            
            # Process batch
            processed_batch = process_batch(batch_df, model, tokenizer, device)
            df.iloc[i:end_idx] = processed_batch
            
            # Save progress every 1000 rows
            if (end_idx % 1000) < batch_size:
                temp_file = f'translated_categories_temp_{end_idx}.xlsx'
                df.to_excel(temp_file, index=False)
                print(f"\nSaved progress at {end_idx} rows")
            
            # Memory management
            if get_available_memory() < 1.0:  # Less than 1GB available
                print("\nLow memory detected, cleaning up...")
                gc.collect()
        
        # Save the final results
        output_file = 'translated_categories.xlsx'
        print(f"\nSaving final results to {output_file}...")
        df.to_excel(output_file, index=False)
        print("Translation complete!")
        
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        cleanup_temp_files()

if __name__ == "__main__":
    main() 