import struct

file = "1002.pot"


with open(file, "rb") as f:
    for _ in range(0, 0):
        f.read(struct.unpack("h", f.read(2))[0] - 2)
    total = 0
    while True:

        _sample_size = f.read(2)
        if len(_sample_size) == 0:
            break
        sample_size = struct.unpack("h", _sample_size)[0]
      #  tag_code = struct.unpack("l", f.read(4))[0]
        _tag_code = f.read(2)
        ba = bytearray(_tag_code)
        b2 = ba[1]
        ba[1] = ba[0]
        ba[0] = b2

        f.read(2)
        try:
            tag_code = ba.decode("gb18030")
        except Exception:
            print(str(total) + " gb2312 decode exception")
        else:
            print(str(total) + " " + tag_code)

        stroke_number = struct.unpack("h", f.read(2))[0]
        strokex = []
        strokey = []
        for _ in range(0, stroke_number):
            sx = []
            sy = []
            while True:
                px = struct.unpack("h", f.read(2))[0]
                py = struct.unpack("h", f.read(2))[0]
                if px == -1 and py == 0:
                    break
                sx.append(px)
                sy.append(py)
            strokex.append(sx)
            strokey.append(sy)
        if struct.unpack("l", f.read(4))[0] != -1:
            print("error")
        total += 1
