
# clean_gutenberg.py
import os
import chardet

source_dir = "data/raw"
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
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text

# Open the output file once in append mode
with open(output_file, "w", encoding="utf-8") as out_f:
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

        cleaned = clean_text(text)
        if len(cleaned) < 500:  # skip tiny/empty files
            print(f"⚠️ Skipping {filename}, too short.")
            continue

        # Add separator between books
        out_f.write(cleaned + "\n\n=== NEW BOOK ===\n\n")
        print(f"✅ Added {filename}")

print("🎉 All files cleaned and appended to training.txt")