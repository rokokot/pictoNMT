#!/usr/bin/env python3
import os
import sys
import json
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter

def process_propicto_files(input_dir, output_file):
    print(f"Scanning directory: {input_dir}")
    
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    combined_data = []
    
    sentence_lengths = []
    picto_token_lengths = []
    picto_sequence_lengths = []
    all_pictograms = []  # List to store all pictogram IDs
    
    for json_file in tqdm(json_files, desc="Processing files"):
        file_basename = os.path.basename(json_file).replace(".json", "")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data structures
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = list(data.values())
            else:
                print(f"Unexpected data format in {os.path.basename(json_file)}")
                continue
            
            # Process each item
            for item in items:
                # Skip items missing required fields
                if "sentence" not in item or "pictos" not in item:
                    continue
                
                # Get pictogram tokens if available
                picto_tokens = item.get("pictos_tokens", "")
                
                # Create formatted item
                formatted_item = {
                    "source_file": file_basename,
                    "pictogram_sequence": item["pictos"],
                    "target_text": item["sentence"]
                }
                
                # Add pictogram tokens if available
                if picto_tokens:
                    formatted_item["pictogram_tokens"] = picto_tokens
                
                combined_data.append(formatted_item)
                
                # Update statistics
                sentence_lengths.append(len(item["sentence"].split()))
                if picto_tokens:
                    if isinstance(picto_tokens, str):
                        picto_token_lengths.append(len(picto_tokens.split()))
                    elif isinstance(picto_tokens, list):
                        picto_token_lengths.append(len(picto_tokens))
                picto_sequence_lengths.append(len(item["pictos"]))
                
                # Add all pictograms to the counter list
                all_pictograms.extend(item["pictos"])
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    if not combined_data:
        print("No valid data found.")
        return
    
    # Save the combined dataset
    print(f"Saving {len(combined_data)} entries to {output_file}")
    
    # First, save with normal formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    # Now read the file and fix the pictogram sequence formatting
    print("Reformatting pictogram sequences to single line...")
    with open(output_file, 'r', encoding='utf-8') as f:
        json_content = f.read()
    
    import re
    pattern = r'"pictogram_sequence": \[\s+([^]]+?)\s+\]'
    replacement = lambda m: '"pictogram_sequence": [' + re.sub(r'\s+', ' ', m.group(1)) + ']'
    single_line_json = re.sub(pattern, replacement, json_content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(single_line_json)
    
    stats_output_file = output_file.replace(".json", "_stats.json")
    
    pictogram_counter = Counter(all_pictograms)
    
    stats = {
        "dataset": {
            "total_entries": len(combined_data),
            "unique_sentences": len(set(item['target_text'] for item in combined_data)),
            "source_files": len(set(item['source_file'] for item in combined_data)),
            "source_file_counts": dict(Counter(item['source_file'] for item in combined_data))
        },
        "sentences": {
            "average_length": float(np.mean(sentence_lengths)),
            "min_length": min(sentence_lengths),
            "max_length": max(sentence_lengths),
            "length_distribution": dict(Counter(sentence_lengths))
        },
        "pictogram_sequences": {
            "average_length": float(np.mean(picto_sequence_lengths)),
            "min_length": min(picto_sequence_lengths),
            "max_length": max(picto_sequence_lengths),
            "length_distribution": dict(Counter(picto_sequence_lengths))
        },
        "pictograms": {
            "total_occurrences": len(all_pictograms),
            "unique_pictograms": len(pictogram_counter),
            "top_50_pictograms": dict(pictogram_counter.most_common(50))
        }
    }
    
    # Save statistics to JSON file
    print(f"Saving statistics to {stats_output_file}")
    with open(stats_output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total entries: {len(combined_data)}")
    print(f"Number of unique sentences: {len(set(item['target_text'] for item in combined_data))}")
    print(f"Source files: {len(set(item['source_file'] for item in combined_data))}")
    
    if sentence_lengths:
        print(f"Average sentence length: {np.mean(sentence_lengths):.2f} words")
        print(f"Min/Max sentence length: {min(sentence_lengths)}/{max(sentence_lengths)} words")
    
    if picto_sequence_lengths:
        print(f"Average pictogram sequence length: {np.mean(picto_sequence_lengths):.2f} pictograms")
        print(f"Min/Max pictogram sequence length: {min(picto_sequence_lengths)}/{max(picto_sequence_lengths)} pictograms")
    
    # Count and display unique pictograms
    pictogram_counter = Counter(all_pictograms)
    print(f"Total pictogram occurrences: {len(all_pictograms)}")
    print(f"Number of unique pictograms: {len(pictogram_counter)}")
    
    # Show most common pictograms
    print("\nTop 10 most common pictograms:")
    for pictogram, count in pictogram_counter.most_common(10):
        print(f"  Pictogram ID {pictogram}: {count} occurrences")
        
    print(f"\nStatistics saved to {stats_output_file}")


if __name__ == "__main__":
    
    input_dir = "./data/propicto-source"
    
    # Take output file from command line or use default
  
    output_file = "./data/propicto_combined.json"
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    
    # Process files
    process_propicto_files(input_dir, output_file)