
# clean_gutenberg.py
import os
import chardet

input_file = "data/raw/10084-8.txt"
output_file = "data/training.txt"    # cleaned output

with open(input_file, "rb") as f:
    raw = f.read()
result = chardet.detect(raw)
encoding = result["encoding"]

print(f"Detected encoding: {encoding}")

text = raw.decode(encoding)

# Find the typical Gutenberg start and end markers
start_idx = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
if start_idx != -1:
    text = text[start_idx - 1:]
    
end_idx = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
if end_idx != -1:
    text = text[:end_idx]

# Optionally remove extra line breaks
text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

# Save cleaned file
with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Cleaned text saved to {output_file}")