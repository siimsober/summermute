# clean_wikipedia.py
import bz2
import html
import re
import sys
import xml.etree.ElementTree as ET
from tokenizer import EOT_BYTE

if len(sys.argv) < 2 or sys.argv[1] not in ("est", "eng"):
    print("Usage: python clean_wikipedia.py <est|eng>")
    sys.exit(1)
lang = sys.argv[1]

# est: Estonian Wikipedia, eng: Simple English Wikipedia
SOURCES = {
    "est": "data/raw/etwiki-latest-pages-articles-xml/etwiki-latest-pages-articles.xml",
    "eng": "data/raw/simplewiki/simplewiki-latest-pages-articles.xml.bz2",
}
# Sections that are mostly references and lists, cut from here to the end
TAIL_SECTION_NAMES = {
    "est": "Viited|Märkused|Välislingid|Kirjandus|Vaata ka|Allikad",
    "eng": "References|Notes|Other websites|External links|Related pages|See also|Sources|Further reading",
}
# Disambiguation page templates, those pages are skipped
DISAMBIG = {
    "est": ("{{täpsustus",),
    "eng": ("{{disambig", "{{dab", "{{surname", "{{given name"),
}

source_file = SOURCES[lang]
output_file = f"data/training-{lang}.txt"
MAX_BYTES = 12_000_000   # stop once this much clean text is written
MIN_ARTICLE_LEN = 200    # skip stubs shorter than this
MIN_SECTION = 300        # min chars between EOTs, short sections are merged into the next one
EOT = EOT_BYTE.decode()
SECTION = "\x00"         # temporary marker for top-level (==) section headings

# Links into these namespaces are dropped entirely (files, categories)
DROP_LINK_PREFIX = re.compile(
    r"^\s*(file|image|fail|pilt|category|kategooria|media|meedia)\s*:", re.I)
# Interlanguage links like [[en:Foo]]
LANG_LINK = re.compile(r"^\s*[a-z]{2,3}(-[a-z]+)?\s*:", re.I)
TAIL_SECTIONS = re.compile(
    rf"^=+\s*({TAIL_SECTION_NAMES[lang]})\s*=+\s*$", re.I | re.M)
# Tags removed together with their content
DROP_TAGS = re.compile(
    r"<(ref|gallery|math|timeline|syntaxhighlight|source|score|graph|mapframe|imagemap)\b[^>]*?"
    r"(/>|>.*?</\1\s*>)", re.I | re.S)

def remove_nested(text, open_s, close_s):
    """Remove innermost open_s ... close_s blocks repeatedly to handle nesting."""
    pattern = re.compile(re.escape(open_s) + r"(?:(?!" + re.escape(open_s) + r").)*?" + re.escape(close_s), re.S)
    while True:
        new = pattern.sub("", text)
        if new == text:
            return text
        text = new

def replace_link(m):
    inner = m.group(1)
    if DROP_LINK_PREFIX.match(inner) or LANG_LINK.match(inner):
        return ""
    return inner.split("|")[-1]   # [[Target|Text]] -> Text, [[Word]] -> Word

def clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    m = TAIL_SECTIONS.search(text)
    if m:
        text = text[:m.start()]

    text = DROP_TAGS.sub("", text)
    text = remove_nested(text, "{{", "}}")    # templates, infoboxes
    text = remove_nested(text, "{|", "|}")    # tables
    text = re.sub(r"<[^>]+>", "", text)       # remaining html tags, keep content

    # Internal links, innermost first so file captions with links are handled
    link = re.compile(r"\[\[([^\[\]]*)\]\]")
    while True:
        new = link.sub(replace_link, text)
        if new == text:
            break
        text = new

    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)  # [url text] -> text
    text = re.sub(r"\[https?://[^\]]*\]", "", text)                 # bare [url]
    text = re.sub(r"'{2,}", "", text)                               # bold / italic
    text = re.sub(r"^==\s*([^=].*?)\s*==\s*$", SECTION + r"\1", text, flags=re.M)  # sections
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.M) # subheadings -> plain line
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.M)             # list markers
    text = re.sub(r"__[A-Z]+__", "", text)                          # magic words like __TOC__

    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ([.,;:!?)])", r"\1", text)   # space left before punctuation by removed refs
    text = text.replace("()", "")

    lines = []
    since_eot = 0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(SECTION):
            # EOT before a section heading, unless the text since the last EOT is too short
            line = line[1:].strip()
            if line and since_eot >= MIN_SECTION:
                line = EOT + line
                since_eot = 0
        if not line or line[0] in "|!{}[]":       # leftover table / template junk
            continue
        lines.append(line)
        since_eot += len(line) + 1
    return "\n".join(lines)

def local(tag):
    return tag.rsplit("}", 1)[-1]

written = 0
articles = 0
with open(output_file, "w", encoding="utf-8", newline="\n") as out_f:
    source = bz2.open(source_file, "rb") if source_file.endswith(".bz2") else open(source_file, "rb")
    for event, elem in ET.iterparse(source, events=("end",)):
        if local(elem.tag) != "page":
            continue

        fields = {local(c.tag): c for c in elem}
        ns = fields["ns"].text if "ns" in fields else None
        revision = fields.get("revision")
        text_el = None
        if revision is not None:
            text_el = next((c for c in revision if local(c.tag) == "text"), None)
        raw = text_el.text if text_el is not None and text_el.text else ""
        title = fields["title"].text if "title" in fields else ""
        elem.clear()

        # Only real articles: main namespace, no redirects, no disambiguation pages
        if ns != "0" or "redirect" in fields or raw.lstrip().lower().startswith(("#redirect", "#suuna")):
            continue
        if any(t in raw.lower() for t in DISAMBIG[lang]):
            continue

        clean = clean_wikitext(raw)
        if len(clean) < MIN_ARTICLE_LEN:
            continue

        chunk = f"{title}\n{clean}\n{EOT}"   # EOT after each article
        out_f.write(chunk)
        written += len(chunk.encode("utf-8"))
        articles += 1
        if articles % 1000 == 0:
            print(f"✅ {articles} articles, {written / 1e6:.1f} MB")
        if written >= MAX_BYTES:
            break

print(f"🎉 Wrote {articles} articles ({written / 1e6:.1f} MB) to {output_file}")
