import os
import re

RAW_DATA_DIR = "Raw Data"
OUTPUT_FILE = "data/starwars.txt"
pattern = r'^"\d+"\s+"([A-Z0-9 _\-]+)"\s+"(.+)"$'

def clean_name_dialogue_text(text):
    # Match lines like: NAME" "Dialogue"
    match = re.match(pattern, text.strip())
    if match:
        speaker = match.group(1).strip()
        dialogue = match.group(2).strip()
        clean_text = f'{speaker} : {dialogue}'
        return clean_text
    else:
        return text.strip()

def clean_multi_whitespace_line(text):
    result = re.sub(r'\s{2,}', ' : ', text).strip()
    return result

def get_cleaned_text(file_path, file_type=''):
    cleaned_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != "":
                # Remove non-printable characters and extra spaces
                text = re.sub(r'[^\x20-\x7E\n]', ' ', line)

                if file_type == 'dialogue':
                    cleaned_line = clean_name_dialogue_text(text)
                elif file_type == 'multispace':
                    cleaned_line = clean_multi_whitespace_line(text)
                else :  # Only add non-empty lines
                    cleaned_line = re.sub(r": ", " : ", text).strip()
                
                cleaned_lines.append(cleaned_line)
    return cleaned_lines

def process_raw_data():
    folder = 'Raw Data'
    
    multispace_files = ['EpisodeIV_dialogues.txt', 'EpisodeVI_dialogues.txt']
    files_to_process = ['SW_EpisodeIV.txt', 'SW_EpisodeV.txt', 'SW_EpisodeVI.txt']
    
    all_text = []

    for file_name in files_to_process:
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            clean_lines = get_cleaned_text(file_path, file_type='dialogue')
            all_text.extend(clean_lines)
    
    for file_name in multispace_files:
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            clean_lines = get_cleaned_text(file_path, file_type='multispace')
            all_text.extend(clean_lines)

    file_path = os.path.join(folder, 'EpisodeV_dialogues.txt')
    if os.path.exists(file_path):
        clean_text = get_cleaned_text(file_path)
        all_text.extend(clean_text)

    # Join all cleaned texts with newlines
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join([line.strip() for line in all_text if line.strip() != '']))

if __name__ == "__main__":
    process_raw_data()