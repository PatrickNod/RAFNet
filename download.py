#!/usr/bin/env python3
"""
Simple downloader for the provided AI-Studio URL.

Usage:
  python download_pan.py [URL] [output_path]

If no URL is given, the script uses the URL embedded in the file.
"""
import sys
import os
from urllib.parse import unquote, urlparse, parse_qs

# 修复：将URL拼接为完整的单行字符串，移除多余换行和空格
URL_DEFAULT = (
    "https://ai-studio-online.bj.bcebos.com/v1/a57ce1ab10f94744bb859c409928fe50b759405a51354512a7a67ee796aef8bd"
    "?responseContentDisposition=attachment%3Bfilename%3DPan.zip&authorization=bce-auth-v1%2FALTAKzReLNvew3ySINYJ0fuAMN%2F2026-01-09T11%3A51%3A46Z%2F60%2F%2F1bf18c9cd57dacf61b40580493de7bd2a921ed09772c18ad7e16686b933a3b73"
)


def filename_from_cd(cd):
    # Very small parser for Content-Disposition header
    if not cd:
        return None
    for part in cd.split(';'):
        part = part.strip()
        if part.lower().startswith('filename='):
            val = part.split('=', 1)[1]
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            return val
    return None


def filename_from_url(url):
    qs = parse_qs(urlparse(url).query)
    # responseContentDisposition may be percent-encoded
    rcd = qs.get('responseContentDisposition') or qs.get('responsecontentdisposition')
    if rcd:
        try:
            # extract filename=... from the header value
            decoded = unquote(rcd[0])
            # look for filename= after decoded
            if 'filename=' in decoded:
                return decoded.split('filename=')[-1].strip('"')
        except Exception:
            pass
    fn = qs.get('filename')
    if fn:
        return fn[0]
    return 'Pan.zip'


def download(url, out_path=None):
    try:
        import requests
    except Exception:
        print('requests library not found. Install with: pip install requests')
        return 2

    if out_path is None:
        out_path = filename_from_url(url)

    # Stream the download
    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
        except Exception as e:
            print('Download failed:', e)
            return 3

        cd = r.headers.get('content-disposition')
        fn = filename_from_cd(cd) or out_path
        # if out_path is a directory, append filename
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, fn)

        total = r.headers.get('content-length')
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None

        print(f'Saving to: {out_path}')
        downloaded = 0
        chunk_size = 1024 * 32
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f'\r{downloaded}/{total} bytes ({pct:.1f}%)', end='', flush=True)
        if total:
            print('\nDownload complete')
        else:
            print('Download complete (size unknown)')

    return 0


def main(argv):
    url = URL_DEFAULT
    out = None
    if len(argv) >= 2:
        url = argv[1]
    if len(argv) >= 3:
        out = argv[2]

    return download(url, out)


if __name__ == '__main__':
    sys.exit(main(sys.argv))