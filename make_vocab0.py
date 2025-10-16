#make_vocab0.py

import csv

# -----------------------------
# Control character names (C0 + DEL)
# -----------------------------
CONTROL_NAMES = {
    0: "NUL", 1: "SOH", 2: "STX", 3: "ETX", 4: "EOT", 5: "ENQ", 6: "ACK",
    7: "BEL", 8: "BS", 9: "TAB", 10: "LF", 11: "VT", 12: "FF", 13: "CR",
    14: "SO", 15: "SI", 16: "DLE", 17: "DC1", 18: "DC2", 19: "DC3",
    20: "DC4", 21: "NAK", 22: "SYN", 23: "ETB", 24: "CAN", 25: "EM",
    26: "SUB", 27: "ESC", 28: "FS", 29: "GS", 30: "RS", 31: "US", 127: "DEL"
}

# -----------------------------
# Human-readable representation
# -----------------------------
def readable_repr(byte_seq):
    """Return a readable string representation of a list of bytes."""
    parts = []
    for b in byte_seq:
        if b in CONTROL_NAMES:
            parts.append(f"<{CONTROL_NAMES[b]}>")
        elif 33 <= b <= 126:
            parts.append(chr(b))
        else:
            # Try to decode as UTF-8 if part of valid multi-byte char
            try:
                s = bytes([b]).decode("utf-8")
                parts.append(s)
            except UnicodeDecodeError:
                parts.append(f"<0x{b:02X}>")
    return "".join(parts)

# -----------------------------
# Generate vocab-0.tsv
# -----------------------------
with open("data/vocab/vocab-0.tsv", "w", newline="", encoding="utf-8") as f:
    f.write("id\tbytes\trepr\n")
    for i in range(256):
        byte_list = [i]
        byte_str = ",".join(map(str, byte_list))
        repr_str = f"[{readable_repr(byte_list)}]"
        f.write(f"{i}\t{byte_str}\t{repr_str}\n")

print("✅ Wrote vocab-0.tsv with 256 base byte tokens")