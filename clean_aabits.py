# clean_aabits.py
# Simplified Estonian first reader (aabits) PDF -> training text.
# The book is all capitals with compound words split by hyphens for young readers
# (HEINA-MAA, SÜNNI-PÄEV). This joins the words and converts to normal sentence case.
import os
import re
import shutil
import subprocess
from tokenizer import EOT_BYTE

source_file = "data/raw/aabits.pdf"
output_file = "data/aabits-est.txt"   # "-est" so s_train.py uses the Estonian vocab
EOT = EOT_BYTE.decode()

# Names in the book, so they keep their capital letter. Inflected forms (JAAGUPILE, TEELEL)
# are matched by NAME_ENDINGS. Ordinary words that look like names are left out on purpose:
# ANNA (give), PEETE (beets), KALLI (hug), TIPP, URR.
NAMES = {
    # people
    "Jaagup", "Kaspar", "Artur", "Teele", "Oskar", "Heino", "Piia", "Betti", "Siim", "Jürgen",
    "Mart", "Mardi", "Karin", "Katja", "Jekaterina", "Fedja", "Ülo", "Ülle", "Pille", "Kadri",
    "Linda", "Sohvi", "Paula", "Filipp", "Anu", "Marek", "Ott", "Ave", "Ain", "Rain", "Kaie",
    "Olav", "Mati", "Jüri", "Juku", "Juhan", "Gustav", "Ernesaks", "Robin", "Kaupo", "Mari",
    "Anne", "Cipollino", "Cipollone", "Siimu",
    # animals
    "Timm",
    # places and brands
    "Eesti", "Pärnu", "Pariis", "Ukraina", "Aasia", "Filipiinid", "Gröönimaa", "Lapimaa",
    "Galapagos", "Piusa", "Audi", "Škoda",
}
NAME_ENDINGS = {"", "i", "t", "l", "le", "lt", "ga", "st", "s", "sse", "ni", "ks", "il", "ile",
                "ilt", "iga", "ist", "is", "isse", "iks", "de", "dele", "dest"}


def pdf_to_text(path):
    """Extract text with pdftotext (poppler). Git Bash ships it in mingw64/bin."""
    exe = shutil.which("pdftotext") or r"C:\Program Files\Git\mingw64\bin\pdftotext.exe"
    if not os.path.exists(exe) and shutil.which(exe) is None:
        raise SystemExit("❌ pdftotext not found (comes with Git Bash or poppler)")
    return subprocess.run([exe, "-enc", "UTF-8", path, "-"], capture_output=True,
                          check=True).stdout.decode("utf-8")


def is_name(word):
    w = word.capitalize()
    return any(w.startswith(n) and w[len(n):] in NAME_ENDINGS for n in NAMES)


def sentence_case(text):
    text = text.lower()
    # names keep their capital
    text = re.sub(r"[^\W\d_]+", lambda m: m.group(0).capitalize() if is_name(m.group(0)) else m.group(0), text)
    # capital at the start, after . ! ? … and at the start of direct speech „...“
    # (not after !“ — the speech tag that follows is lowercase: „See on ju puravik!“ hüüab Kaspar.)
    return re.sub(r"(^|[.!?…]\s+|„)(„?)([^\W\d_])",
                  lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)


raw = pdf_to_text(source_file).replace("\f", "\n")

# Words split across line ends (MAHE-\nTALU), then compound hyphens inside words (HEINA-MAA).
# Dashes with spaces around them (KALLI – KALLI) are punctuation and stay.
text = re.sub(r"(?<=[^\W\d_])-\n(?=[^\W\d_])", "", raw)
text = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", "", text)

# Stories are separated by blank lines; lines inside a story are joined into one paragraph.
stories = [" ".join(line.strip() for line in block.splitlines() if line.strip())
           for block in re.split(r"\n\s*\n", text)]
stories = [s for s in stories if s][1:]   # first block is the title page (authors, title)

stories = [re.sub(r"\s+", " ", sentence_case(s)).strip() for s in stories]

with open(output_file, "w", encoding="utf-8", newline="\n") as f:
    for story in stories:
        f.write(story + "\n" + EOT)   # EOT after each story, like the Wikipedia sections

print(f"🎉 Wrote {len(stories)} stories, {sum(len(s.split()) for s in stories)} words to {output_file}")
