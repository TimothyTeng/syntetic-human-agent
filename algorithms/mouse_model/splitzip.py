"""
Minimal reader for *split* ZIP archives (name.z01, name.z02, ..., name.zip).

Python's zipfile and Windows' built-in tools can't open multi-part archives,
and the BMDD download is one (~4.8 GB in 10 parts). This reader parses the
central directory (including ZIP64 records), then reads and inflates single
members on demand, so sessions can be streamed straight from the archive
without extracting everything to disk.

CLI (from project root):
    python -m algorithms.mouse_model.splitzip list    boun-mouse-dynamics-dataset.zip
    python -m algorithms.mouse_model.splitzip extract boun-mouse-dynamics-dataset.zip --pattern training --dest data/bmdd --max-files 200
"""

import argparse
import collections
import fnmatch
import os
import struct
import zlib
from dataclasses import dataclass


@dataclass
class Member:
    name: str
    method: int        # 0 = stored, 8 = deflate
    csize: int         # compressed size
    usize: int         # uncompressed size
    disk: int          # part index where the local header starts
    offset: int        # local header offset within that part


class SplitZip:
    def __init__(self, zip_path):
        """zip_path: path of the final '.zip' part; '.z01'... are found next to it."""
        base, _ = os.path.splitext(zip_path)
        self.parts = []
        i = 1
        while os.path.exists(f"{base}.z{i:02d}"):
            self.parts.append(f"{base}.z{i:02d}")
            i += 1
        self.parts.append(zip_path)
        self.sizes = [os.path.getsize(p) for p in self.parts]
        self._handles = {}
        self.members = self._read_central_directory()

    # --- Low-level reading ------------------------------------------------------

    def _fh(self, disk):
        if disk not in self._handles:
            self._handles[disk] = open(self.parts[disk], "rb")
        return self._handles[disk]

    def _read(self, disk, offset, n):
        """Read n bytes starting at (disk, offset), continuing into later parts."""
        out = bytearray()
        while n > 0:
            if offset >= self.sizes[disk]:
                disk, offset = disk + 1, 0
                continue
            f = self._fh(disk)
            f.seek(offset)
            chunk = f.read(min(n, self.sizes[disk] - offset))
            out += chunk
            n -= len(chunk)
            offset += len(chunk)
        return bytes(out), disk, offset

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    # --- Central directory -----------------------------------------------------------

    def _read_central_directory(self):
        last = len(self.parts) - 1
        tail_len = min(self.sizes[last], 1 << 17)
        tail, _, _ = self._read(last, self.sizes[last] - tail_len, tail_len)

        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError("End of central directory not found - is this the last '.zip' part?")
        _, _, cd_disk, _, n_total, cd_size, cd_off, _ = struct.unpack("<4s4H2LH", tail[i:i + 22])

        k = tail.rfind(b"PK\x06\x06")  # ZIP64 end record (large archives)
        if k >= 0:
            v = struct.unpack("<4sQHHLLQQQQ", tail[k:k + 56])
            cd_disk, n_total, cd_size, cd_off = v[5], v[7], v[8], v[9]

        cd, _, _ = self._read(cd_disk, cd_off, cd_size)
        members, p = [], 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (_, _, flag, meth, _, _, _, cs, us, nl, el, cl, disk, _, _, off) = \
                struct.unpack("<6H3L5HLL", cd[p + 4:p + 46])
            raw_name = cd[p + 46:p + 46 + nl]
            name = raw_name.decode("utf-8" if flag & 0x800 else "cp437", "replace")
            us, cs, off, disk = _apply_zip64(cd[p + 46 + nl:p + 46 + nl + el], us, cs, off, disk)
            members.append(Member(name, meth, cs, us, disk, off))
            p += 46 + nl + el + cl
        return members

    # --- Member access ----------------------------------------------------------------

    def names(self, pattern=None):
        """Member names, optionally filtered by a glob ('*training*.csv') or substring."""
        out = [m.name for m in self.members if not m.name.endswith("/")]
        if pattern:
            if any(ch in pattern for ch in "*?["):
                out = [n for n in out if fnmatch.fnmatch(n.lower(), pattern.lower())]
            else:
                out = [n for n in out if pattern.lower() in n.lower()]
        return out

    def read(self, name):
        """Return the uncompressed bytes of one member."""
        m = self._by_name().get(name)
        if m is None:
            raise KeyError(name)
        header, disk, off = self._read(m.disk, m.offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Bad local header for {name}")
        nl, el = struct.unpack("<HH", header[26:30])
        _, disk, off = self._read(disk, off, nl + el)       # skip name + extra
        data, _, _ = self._read(disk, off, m.csize)
        if m.method == 0:
            return data
        if m.method == 8:
            return zlib.decompress(data, -15)
        raise NotImplementedError(f"Compression method {m.method} not supported ({name})")

    def extract(self, name, dest_root):
        """Write one member under dest_root, keeping its folder structure."""
        path = os.path.join(dest_root, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.read(name))
        return path

    def _by_name(self):
        if not hasattr(self, "_index"):
            self._index = {m.name: m for m in self.members}
        return self._index

    def summary(self, depth=3):
        """Counts / sizes per top-level folder (for a quick look at the layout)."""
        groups = collections.defaultdict(lambda: [0, 0])
        for m in self.members:
            key = "/".join(m.name.split("/")[:depth])
            groups[key][0] += 1
            groups[key][1] += m.usize
        return groups


def _apply_zip64(extra, us, cs, off, disk):
    """Replace 0xFFFFFFFF placeholders with values from the ZIP64 extra field."""
    q = 0
    while q + 4 <= len(extra):
        hid, hl = struct.unpack("<HH", extra[q:q + 4])
        if hid == 0x0001:
            body, r = extra[q + 4:q + 4 + hl], 0
            if us == 0xFFFFFFFF:
                us = int.from_bytes(body[r:r + 8], "little"); r += 8
            if cs == 0xFFFFFFFF:
                cs = int.from_bytes(body[r:r + 8], "little"); r += 8
            if off == 0xFFFFFFFF:
                off = int.from_bytes(body[r:r + 8], "little"); r += 8
            if disk == 0xFFFF:
                disk = int.from_bytes(body[r:r + 4], "little")
        q += 4 + hl
    return us, cs, off, disk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "extract"])
    ap.add_argument("archive", help="the final .zip part")
    ap.add_argument("--pattern", default=None, help="glob or substring filter on member names")
    ap.add_argument("--dest", default="data/bmdd")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--depth", type=int, default=3, help="folder depth for 'list' summary")
    args = ap.parse_args(argv)

    z = SplitZip(args.archive)
    print(f"{len(z.parts)} parts, {len(z.members)} members, "
          f"{sum(m.usize for m in z.members) / 1e9:.1f} GB uncompressed, "
          f"methods {dict(collections.Counter(m.method for m in z.members))}")
    if args.command == "list":
        for key, (n, size) in sorted(z.summary(args.depth).items()):
            print(f"{n:>8} files {size / 1e6:>10.1f} MB  {key}")
        for name in z.names(args.pattern)[:5]:
            print("  e.g.", name)
    else:
        names = z.names(args.pattern)[:args.max_files]
        for i, name in enumerate(names, 1):
            z.extract(name, args.dest)
            if i % 100 == 0 or i == len(names):
                print(f"  extracted {i}/{len(names)}")
    z.close()


if __name__ == "__main__":
    main()
