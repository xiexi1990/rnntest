import struct

file = "1002.pot"

with open(file, "rb") as f:
    for _ in range(0, 90):
        f.read(struct.unpack("h", f.read(2))[0] - 2)
    with open("1002_part.pot", "wb") as fout:
        fout.write(f.read())
        fout.close()
        f.close()
