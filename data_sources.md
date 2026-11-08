# Data sources

What text the project uses or has downloaded, how much of it is used, and how much more exists.
Raw downloads live in `data/raw/` (not in git). Training files are made by the `clean_*.py` scripts.

## In use for training

| Source | Language | Downloaded | Used | More available | Licence | Script |
|---|---|---|---|---|---|---|
| Estonian Wikipedia dump (`etwiki-latest-pages-articles.xml`) | est | 1.2 GB XML | first 12 MB of clean text (~2,000 articles) -> `data/training-est.txt` | ~100x more: raise `MAX_BYTES` | CC BY-SA | `clean_wikipedia.py est` |
| Simple English Wikipedia dump (`simplewiki-latest-pages-articles.xml.bz2`, 2026-11-02) | eng | 360 MB bz2 | first 12 MB (~2,600 articles) -> start of `data/training-eng.txt` | ~25x more | CC BY-SA | `clean_wikipedia.py eng` |
| Project Gutenberg, 40 adult books (`data/raw/gutenberg/`) | eng | 40 books | all, 11.5 MB -> end of `data/training-eng.txt` | ~70,000 English books | public domain (US) | `clean_gutenberg.py` |
| Aabits, simplified first reader (Vaiksoo, Toomla, Ilves; simplified by Kadrian Hiie) | est | 2 PDFs | 57 stories, 36 KB -> `data/aabits-est.txt` | — | copyrighted: private use only | `clean_aabits.py` |

## Downloaded, not used yet

| Source | Language | Downloaded | More available | Licence | Notes |
|---|---|---|---|---|---|
| **TinyStories V2** (`roneneldan/TinyStories` on Hugging Face) | eng | train 2.23 GB (~2.7M stories) + valid 22.5 MB | V1 (GPT-3.5 + GPT-4) 1.9 GB, TinyStories-Instruct | CDLA-Sharing-1.0 | very simple stories made for tiny models; separated by `<\|endoftext\|>` |
| **Gutenberg children's books** (`data/raw/gutenberg-children/`) | eng | **~100 MB** (first books in catalog order) | **~6,930 English children's / YA books, ~2 GB** in total; also 152 Finnish, 104 German, 75 French | public domain (US) | `download_gutenberg_children.py --max-mb N` (resumable; `--max-mb 0` = all). `children_books.csv` lists all of them with bookshelves. Mostly 1850–1925 language |
| **Alpaca-est** (`TartuNLP/alpaca-est` on GitHub) | est | `alpaca_est.json`, 28.9 MB, 52,006 instruction / input / output examples | — | **none stated**: private use only | generated with gpt-3.5-turbo; labels Juhis / Sisend / Väljund |
| Gutenberg catalog (`pg_catalog.csv`) | — | 21 MB | updated weekly | — | metadata of all ~80,000 Gutenberg books: language, subjects, bookshelves |

## Known, not downloaded

| Source | Language | Size | Licence | Notes |
|---|---|---|---|---|
| Estonian National Corpus (eesti keele ühendkorpus) 2021/2023 | est | ~2.4 billion words | academic terms via keeleressursid.ee / Metashare (sign-up) | mostly web text; the Reference Corpus part (news, fiction) is cleaner. Third-party copy: `siimh/estonian_corpus_2021` on Hugging Face |
| Corpus of Estonian fiction (Metashare) | est | ~5.6M words | academic, non-commercial | adult fiction since 1990 |
| CHILDES caretaker speech (Tartu syllabified corpora) | est | ~400k words | academic | adults talking to children; spoken style |
| `saillab/alpaca-estonian-cleaned` (Hugging Face) | est | 52k examples | check (Alpaca-based, likely non-commercial) | English Alpaca machine-translated to Estonian; Alpaca-est is the better Estonian |
| Dolly 15k (Databricks) | eng | 15k pairs | CC BY-SA 3.0 | human-written instruction / response |
| OpenAssistant OASST1 / OASST2 | multi | ~10k conversation trees | Apache 2.0 | human-written, ranked replies |
| Alpaca / Alpaca-cleaned (Stanford) | eng | 52k pairs | CC BY-NC | generated |
| Anthropic HH-RLHF | eng | ~160k dialogues | MIT | preferred / rejected answers |
| FLAN collection (Google) | eng | millions | mixed open | NLP tasks as instructions |

## Rule of thumb for how much is needed

About 20 training tokens per model parameter. At ~2.3 bytes per token:

| Model size | Tokens wanted | Text |
|---|---|---|
| 0.2M params (current) | 4M | ~10 MB |
| 3M params | 60M | ~140 MB |
| 10M params | 200M | ~460 MB |
