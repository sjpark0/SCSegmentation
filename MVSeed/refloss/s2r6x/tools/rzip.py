"""Random-access read of a remote zip via HTTP Range (no full download). Tracks bytes fetched."""
import io, sys, urllib.request, zipfile, json
FETCHED = [0]
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url; self.pos = 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b''
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        FETCHED[0] += len(data); self.pos += len(data)
        return data
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_zip(url):
    f = io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 16)
    return zipfile.ZipFile(f)
