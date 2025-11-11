# tokenizer.py
import numpy as np

class BPETokenizer:
    def __init__(self, vocab_tsv_path):
        # Load vocabulary file (id, bytes, text_repr, freq)
        self.id_to_bytes = {}
        self.bytes_to_id = {}
        with open(vocab_tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "" or line.startswith("id"):
                    continue
                parts = line.strip().split("\t")
                token_id = int(parts[0])
                byte_seq = tuple(int(b) for b in parts[1].split(","))  # comma-separated bytes
                self.id_to_bytes[token_id] = byte_seq
                self.bytes_to_id[byte_seq] = token_id

    def encode(self, data: bytes) -> np.ndarray:
        """Translate input bytes to token IDs using existing vocabulary."""
        token_ids = []
        i = 0
        while i < len(data):
            # Try to match the longest possible token at this position
            matched = False
            # Tokens can be >1 byte, check lengths descending
            for length in range(min(10, len(data)-i), 0, -1):  # max token length = 10, adjust as needed
                seq = tuple(data[i:i+length])
                if seq in self.bytes_to_id:
                    token_ids.append(self.bytes_to_id[seq])
                    i += length
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No token found for byte sequence starting at index {i}: {data[i:i+10]}")
        return token_ids
    
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
    
    def add_token(self, byte_seq):
        """
        Add a new token to the tokenizer.
        :param byte_seq: tuple of bytes for the new token
        :return: new token ID
        """
        new_id = max(self.id_to_bytes.keys()) + 1
        self.id_to_bytes[new_id] = byte_seq
        self.bytes_to_id[byte_seq] = new_id
        #print(f"Added token {new_id}: bytes={byte_seq}, readable={self.readable_repr(byte_seq)}")
        return new_id
    
    def save(self, file):
        """
        Save the current vocabulary to a TSV file with columns:
        id, bytes, readable_repr
        """
        with open(file, "w", encoding="utf-8", newline="") as f:
            f.write("id\tbytes\trepr\n")
            for token_id in sorted(self.id_to_bytes.keys()):
                byte_seq = self.id_to_bytes[token_id]
                byte_str = ",".join(map(str, byte_seq))
                repr_str = f"[{self.readable_repr(byte_seq)}]"
                f.write(f"{token_id}\t{byte_str}\t{repr_str}\n")
        print(f"✅ Wrote {file} with {len(self.id_to_bytes)} tokens")
    
    def get_vocab_size(self):
        return max(self.id_to_bytes.keys()) + 1