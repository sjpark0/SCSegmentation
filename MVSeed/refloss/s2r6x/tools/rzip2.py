"""Remote zip member fetch: central directory once, then one HTTP Range request per member (threaded)."""
import io, struct, urllib.request, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
import rzip
FETCHED = rzip.FETCHED
class RemoteZip:
    def __init__(self, url):
        self.url = url
        self.z = rzip.open_zip(url)
        self.info = {i.filename: i for i in self.z.infolist()}
    def _get(self, name):
        i = self.info[name]
        start = i.header_offset; end = start + 30 + len(i.filename.encode()) + 512 + i.compress_size
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r: buf = r.read(); break
            except Exception:
                if attempt == 3: raise
        FETCHED[0] += len(buf)
        n, m = struct.unpack('<HH', buf[26:30])
        data = buf[30 + n + m: 30 + n + m + i.compress_size]
        if i.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(data, -15)
        elif i.compress_type != zipfile.ZIP_STORED: raise ValueError(i.compress_type)
        return data
    def get_many(self, names, workers=16):
        with ThreadPoolExecutor(workers) as ex:
            return dict(zip(names, ex.map(self._get, names)))
