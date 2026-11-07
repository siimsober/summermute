# tokenizer.py
import re
import numpy as np

# End-of-text marker between books, chapters, articles and sections.
# Byte 0x04 (EOT) never occurs in normal text and is token id 4 in the base vocab.
EOT_BYTE = b"\x04"
EOT_ID = 4

# Text is split into chunks before BPE (like GPT-2/GPT-4) and merges never cross a chunk boundary,
# so a word looks the same wherever it appears: ". This" is "." + " This", never ". T" + "his".
PRETOKENIZE = re.compile(
    r"'(?i:[sdmt]|ll|ve|re)"     # English contractions: 's 't 'm 'd 'll 've 're
    r"| ?[^\W\d_]+"              # word with optional leading space, Unicode letters (õ ä ö ü)
    r"| ?\d{1,3}"                # numbers, at most 3 digits per chunk
    r"| ?(?:[^\s\w\x04]|_)+"     # punctuation with optional leading space
    r"|\s+(?!\S)|\s+"            # whitespace; the last space before a word goes with the word
    r"|\x04"                     # EOT is always alone
)

def chunk_starts(data: bytes) -> np.ndarray:
    """Boolean array over the bytes of data, True where a pre-tokenizer chunk starts."""
    text = data.decode("utf-8")
    arr = np.frombuffer(data, dtype=np.uint8)
    char_to_byte = np.flatnonzero((arr & 0xC0) != 0x80)   # byte offset of each character
    char_starts = np.fromiter((m.start() for m in PRETOKENIZE.finditer(text)), dtype=np.int64)
    starts = np.zeros(len(data), dtype=bool)
    starts[char_to_byte[char_starts]] = True
    return starts

def merge_pair(tokens, starts, a, b, new_id):
    """
    Replace every (a, b) in tokens with new_id, left to right without overlaps,
    skipping pairs where b starts a new chunk.
    :param tokens: 1-D numpy array of token ids (not modified)
    :param starts: bool array, True where a token starts a chunk (not modified)
    :return: (new tokens, new starts, positions in the old array where a merge started)
    """
    pos = np.flatnonzero((tokens[:-1] == a) & (tokens[1:] == b) & ~starts[1:])
    if len(pos) == 0:
        return tokens, starts, pos
    if a == b and len(pos) > 1:
        # In a run like "a a a" the match at i+1 overlaps the one at i: keep every other match per run
        idx = np.arange(len(pos))
        run_start = np.ones(len(pos), dtype=bool)
        run_start[1:] = np.diff(pos) != 1
        start_idx = np.maximum.accumulate(np.where(run_start, idx, 0))
        pos = pos[(idx - start_idx) % 2 == 0]
    keep = np.ones(len(tokens), dtype=bool)
    keep[pos + 1] = False
    out = tokens[keep]
    out[pos - np.arange(len(pos))] = new_id   # each earlier merge shifts later positions left by one
    return out, starts[keep], pos


class BPETokenizer:
    def __init__(self, vocab_tsv_path):
        # Load vocabulary file (id, bytes, repr, merge)
        self.id_to_bytes = {}
        self.bytes_to_id = {}
        self.merges = []    # (a, b, new_id) in the order they were learned
        with open(vocab_tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "" or line.startswith("id"):
                    continue
                parts = line.strip().split("\t")
                token_id = int(parts[0])
                byte_seq = tuple(int(b) for b in parts[1].split(","))  # comma-separated bytes
                self.id_to_bytes[token_id] = byte_seq
                self.bytes_to_id[byte_seq] = token_id
                if len(parts) > 3 and parts[3]:
                    a, b = (int(x) for x in parts[3].split(","))
                    self.merges.append((a, b, token_id))
                elif len(byte_seq) > 1:
                    # Older vocab files have no merge column: find two earlier tokens that make this one
                    self.merges.append(self._find_merge(byte_seq, token_id))
        self.merges.sort(key=lambda m: m[2])

    def _find_merge(self, byte_seq, token_id):
        for i in range(1, len(byte_seq)):
            a = self.bytes_to_id.get(byte_seq[:i])
            b = self.bytes_to_id.get(byte_seq[i:])
            if a is not None and b is not None and a < token_id and b < token_id:
                return (a, b, token_id)
        raise ValueError(f"Token {token_id} {byte_seq} is not a merge of two earlier tokens")

    def encode(self, data: bytes, return_starts=False):
        """
        Translate input bytes to token IDs by applying the merges in the order they were learned,
        within pre-tokenizer chunks. Returns a numpy array (and the chunk starts if return_starts).
        """
        data = bytes(data)
        tokens = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        starts = chunk_starts(data)
        for a, b, new_id in self.merges:
            if len(tokens) < 2:
                break
            tokens, starts, _ = merge_pair(tokens, starts, a, b, new_id)
        return (tokens, starts) if return_starts else tokens
    
    def decode(self, token_ids) -> bytes:
        """Turn token IDs back into raw bytes."""
        out = bytearray()
        for tid in token_ids:
            out.extend(self.id_to_bytes[int(tid)])
        return bytes(out)
    
    @staticmethod
    def readable_repr(byte_seq):
        """Return a readable string representation of a list of bytes."""
        CONTROL_NAMES = {
            0: "NUL", 1: "SOH", 2: "STX", 3: "ETX", 4: "EOT", 5: "ENQ", 6: "ACK",
            7: "BEL", 8: "BS", 9: "TAB", 10: "LF", 11: "VT", 12: "FF", 13: "CR",
            14: "SO", 15: "SI", 16: "DLE", 17: "DC1", 18: "DC2", 19: "DC3",
            20: "DC4", 21: "NAK", 22: "SYN", 23: "ETB", 24: "CAN", 25: "EM",
            26: "SUB", 27: "ESC", 28: "FS", 29: "GS", 30: "RS", 31: "US", 127: "DEL"
        }

        if not isinstance(byte_seq, (bytes, bytearray)):
            byte_seq = bytes(byte_seq)

        try:
            text = byte_seq.decode("utf-8")
        except UnicodeDecodeError:
            text = byte_seq

        parts = []
        for c in text:
            if isinstance(c, str):
                b = ord(c)
                ch = c
            else:
                b = c
                ch = chr(c) if 32 <= c <= 127 else f"<0x{c:02X}>"

            if b in CONTROL_NAMES:
                parts.append(f"<{CONTROL_NAMES[b]}>")
            elif 32 <= b <= 126 or b >= 160:
                parts.append(ch)
            else:
                parts.append(f"<0x{b:02X}>")

        return "".join(parts)
    
    def add_merge(self, a, b):
        """
        Add a new token made by merging tokens a and b.
        :return: new token ID
        """
        new_id = max(self.id_to_bytes.keys()) + 1
        byte_seq = self.id_to_bytes[a] + self.id_to_bytes[b]
        self.id_to_bytes[new_id] = byte_seq
        self.bytes_to_id.setdefault(byte_seq, new_id)
        self.merges.append((a, b, new_id))
        return new_id

    def save(self, file):
        """
        Save the current vocabulary to a TSV file with columns:
        id, bytes, readable_repr, merge (the two token ids it was merged from)
        """
        merge_of = {new_id: (a, b) for a, b, new_id in self.merges}
        with open(file, "w", encoding="utf-8", newline="") as f:
            f.write("id\tbytes\trepr\tmerge\n")
            for token_id in sorted(self.id_to_bytes.keys()):
                byte_seq = self.id_to_bytes[token_id]
                byte_str = ",".join(map(str, byte_seq))
                repr_str = f"[{self.readable_repr(byte_seq)}]"
                merge_str = ",".join(map(str, merge_of[token_id])) if token_id in merge_of else ""
                f.write(f"{token_id}\t{byte_str}\t{repr_str}\t{merge_str}\n")
        print(f"✅ Wrote {file} with {len(self.id_to_bytes)} tokens")
    
    def get_vocab_size(self):
        return max(self.id_to_bytes.keys()) + 1