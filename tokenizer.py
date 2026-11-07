# tokenizer.py
import numpy as np

# End-of-text marker between books, chapters, articles and sections.
# Byte 0x04 (EOT) never occurs in normal text and is token id 4 in the base vocab.
EOT_BYTE = b"\x04"
EOT_ID = 4

def merge_pair(tokens, a, b, new_id):
    """
    Replace every (a, b) in tokens with new_id, left to right without overlaps.
    :param tokens: 1-D numpy array of token ids (not modified)
    :return: (new tokens array, positions in the old array where a merge started)
    """
    pos = np.flatnonzero((tokens[:-1] == a) & (tokens[1:] == b))
    if len(pos) == 0:
        return tokens, pos
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
    return out, pos


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

    def encode(self, data: bytes) -> np.ndarray:
        """Translate input bytes to token IDs by applying the merges in the order they were learned."""
        tokens = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int64)
        for a, b, new_id in self.merges:
            if len(tokens) < 2:
                break
            tokens, _ = merge_pair(tokens, a, b, new_id)
        return tokens
    
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