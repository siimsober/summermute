import re

def clean_wikipedia_xml(text):
    # Remove XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove file/image links
    text = re.sub(r"\[\[(File|Image):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

    # Replace internal links [[Link|Text]] → Text or [[Word]] → Word
    text = re.sub(r"\[\[([^|\]]*\|)?([^\]]+)\]\]", r"\2", text)

    # Remove templates {{...}}
    text = re.sub(r"\{\{[^}]+\}\}", "", text)

    # Remove HTML entities like &amp;
    text = re.sub(r"&[a-z]+;", " ", text)

    # Normalize line breaks: replace multiple newlines with a single newline
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Replace remaining multiple spaces or tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
        
    # Rejoin lines preserving paragraph breaks
    return "\n".join([line for line in lines if line])

with open("data/raw/etwiki-latest-pages-articles-xml/etwiki-latest-pages-articles.xml", encoding="utf-8", errors="ignore") as f:
    raw = f.read(15_000_000)

clean = clean_wikipedia_xml(raw)

with open("data/training-est.txt", "w", encoding="utf-8") as f:
    f.write(clean)