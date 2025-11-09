
# clean_gutenberg.py
import os
import chardet
import re

source_dir = "data/raw/gutenberg/gutenberg_txt"
output_file = "data/training-eng.txt"    # cleaned output

def clean_text(text):
    # Find the typical Gutenberg start and end markers
    start_idx = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
    if start_idx != -1:
        text = text[start_idx + 1:]
    
    end_idx = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
    if end_idx != -1:
        text = text[:end_idx]

    # Optionally remove extra line breaks
    #text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text

def normalize_linebreaks(text: str) -> str:
    # Normalize line endings to just '\n'
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace 2+ line breaks with a placeholder
    text = re.sub(r'\n{2,}', '__PARA__', text)
    # Remove remaining single line breaks
    text = re.sub(r'\n', ' ', text)
    # Restore paragraph breaks
    text = text.replace('__PARA__', '\n')
    
    return text.strip()

# Open the output file once
with open(output_file, "w", encoding="utf-8", newline="\n") as out_f:
    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(".txt"):
            continue

        src_path = os.path.join(source_dir, filename)
        with open(src_path, "rb") as f:
            raw = f.read()

        # Detect encoding
        enc = chardet.detect(raw)["encoding"] or "utf-8"

        try:
            text = raw.decode(enc, errors="ignore")
        except Exception as e:
            print(f"❌ Skipping {filename}, decode failed: {e}")
            continue

        cleaned = normalize_linebreaks(clean_text(text))
        if len(cleaned) < 500:  # skip tiny/empty files
            print(f"⚠️ Skipping {filename}, too short.")
            continue

        # Add separator between books
        out_f.write(cleaned + "\n\n=== NEW BOOK ===\n\n")
        print(f"✅ Added {filename}")

print(f"🎉 All files cleaned and appended to {output_file}")