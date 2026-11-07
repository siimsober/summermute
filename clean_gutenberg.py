
# clean_gutenberg.py
import os
import chardet
import re

source_dir = "data/raw/gutenberg/gutenberg_txt"
output_file = "data/training-eng.txt"    # cleaned output

# Header ends at the end of one of these lines (modern / old "Small Print" etexts)
START_MARKERS = [
    re.compile(r"^\*{2,3} ?START OF (THE|THIS) PROJECT GUTENBERG E(BOOK|TEXT).*$", re.I | re.M),
    re.compile(r"^\*END\*THE SMALL PRINT!.*$", re.I | re.M),
]
# Footer starts at the first of these lines after the header
END_MARKERS = [
    re.compile(r"^\*{2,3} ?END OF (THE|THIS) PROJECT GUTENBERG E(BOOK|TEXT)", re.I | re.M),
    re.compile(r"^\s*End of (the |this )?(Project Gutenberg('s)? )?E-?(book|text)", re.I | re.M),
]
PRODUCED_BY = re.compile(r"^\s*(Produced|Prepared|Transcribed) by[^\n]*(\n[^\n]+)*", re.I)

def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Cut the license header
    for marker in START_MARKERS:
        m = marker.search(text)
        if m:
            text = text[m.end():]
            break

    # Cut the license footer at the earliest end marker
    end_idx = len(text)
    for marker in END_MARKERS:
        m = marker.search(text)
        if m:
            end_idx = min(end_idx, m.start())
    text = text[:end_idx]

    # Drop the "Produced by ..." credits paragraph at the top
    text = PRODUCED_BY.sub("", text.lstrip(), count=1)
    return text

def normalize_linebreaks(text: str) -> str:
    # Normalize line endings to just '\n'
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Replace 2+ line breaks with a placeholder
    text = re.sub(r'\n\s*\n+', '__PARA__', text)
    # Remove remaining single line breaks
    text = re.sub(r'\n', ' ', text)
    # Collapse runs of spaces left by indentation
    text = re.sub(r'[ \t]+', ' ', text)
    # Restore paragraph breaks
    text = text.replace('__PARA__', '\n')
    text = "\n".join(line.strip() for line in text.split('\n'))

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
        if "gutenberg" in cleaned.lower():
            print(f"⚠️ {filename} still mentions Gutenberg after cleaning")

        # Add separator between books
        out_f.write(cleaned + "\n\n=== NEW BOOK ===\n\n")
        print(f"✅ Added {filename}")

print(f"🎉 All files cleaned and appended to {output_file}")
