# download_gutenberg_children.py
# Downloads all English Project Gutenberg books on children's / young adult bookshelves
# from the aleph.gutenberg.org mirror (as Gutenberg asks for bulk downloads), politely:
# one request every 2 seconds. Resumable: books already downloaded are skipped.
# The full set is ~6,900 books / >1 GB; --max-mb stops once that much is on disk (default 100).
import argparse
import csv
import os
import time
import urllib.request
import urllib.error

OUT_DIR = "data/raw/gutenberg-children"
CATALOG = f"{OUT_DIR}/pg_catalog.csv"   # https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
TXT_DIR = f"{OUT_DIR}/txt"
DELAY = 2.0

parser = argparse.ArgumentParser(description="Download Gutenberg children's books")
parser.add_argument("--max-mb", type=float, default=100, help="stop when this many MB are downloaded (0 = all)")
args = parser.parse_args()


def is_children(shelves):
    return any("hild" in s or "Young Adult" in s or "Young People" in s for s in shelves.split(";"))


def mirror_dir(book_id):
    """aleph.gutenberg.org layout: 14838 -> /1/4/8/3/14838/, 11 -> /1/11/, 9 -> /0/9/"""
    s = str(book_id)
    prefix = "/".join(s[:-1]) if len(s) > 1 else "0"
    return f"http://aleph.gutenberg.org/{prefix}/{s}/"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sumerute-children-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


os.makedirs(TXT_DIR, exist_ok=True)
with open(CATALOG, encoding="utf-8") as f:
    books = [r for r in csv.DictReader(f)
             if r["Type"] == "Text" and r["Language"] == "en" and is_children(r["Bookshelves"])]

with open(f"{OUT_DIR}/children_books.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "title", "authors", "bookshelves"])
    for b in books:
        w.writerow([b["Text#"], b["Title"].replace("\n", " "), b["Authors"], b["Bookshelves"]])
print(f"{len(books)} English children's / young adult books in the catalog")

done = failed = skipped = 0
total_bytes = sum(os.path.getsize(os.path.join(TXT_DIR, n)) for n in os.listdir(TXT_DIR))
for i, b in enumerate(books, 1):
    if args.max_mb and total_bytes >= args.max_mb * 1e6:
        print(f"Reached {total_bytes / 1e6:.0f} MB limit (--max-mb {args.max_mb:g}), stopping at book {i}/{len(books)}")
        break
    book_id = b["Text#"]
    out = f"{TXT_DIR}/{book_id}.txt"
    if os.path.exists(out):
        skipped += 1
        continue
    data = None
    # UTF-8 first, then plain ASCII, then Latin-1 (re-encoded to UTF-8)
    for name, enc in ((f"{book_id}-0.txt", "utf-8"), (f"{book_id}.txt", "utf-8"), (f"{book_id}-8.txt", "latin-1")):
        try:
            raw = fetch(mirror_dir(book_id) + name)
            data = raw.decode(enc, errors="replace").encode("utf-8")
            break
        except urllib.error.HTTPError:
            time.sleep(DELAY)
        except Exception as e:
            print(f"⚠️ {book_id}: {e}")
            time.sleep(DELAY * 5)
    if data is None:
        failed += 1
        print(f"❌ {book_id} not found on the mirror: {b['Title'][:60]}")
    else:
        with open(out, "wb") as f:
            f.write(data)
        total_bytes += len(data)
        done += 1
    if i % 50 == 0:
        print(f"{i}/{len(books)}: {done} downloaded, {skipped} already there, {failed} failed, "
              f"{total_bytes / 1e6:.0f} MB")
    time.sleep(DELAY)

print(f"🎉 Finished: {done} downloaded, {skipped} already there, {failed} failed")
