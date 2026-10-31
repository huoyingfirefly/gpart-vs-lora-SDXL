"""Render a report .md next to itself as .html, with relative images kept intact."""

import sys
from pathlib import Path

import markdown

CSS = """
body{max-width:1180px;margin:32px auto;padding:0 24px;
     font:16px/1.68 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:#1a1a1a;background:#fff}
h1{font-size:30px;border-bottom:2px solid #eee;padding-bottom:10px;margin-top:44px}
h2{font-size:23px;border-bottom:1px solid #f0f0f0;padding-bottom:6px;margin-top:36px}
h3{font-size:18px;margin-top:26px}
table{border-collapse:collapse;margin:16px 0;font-size:14px;width:100%}
th,td{border:1px solid #dcdcdc;padding:7px 11px;text-align:left;vertical-align:top}
th{background:#f6f6f6;font-weight:600}
tr:nth-child(even) td{background:#fafafa}
code{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:13px}
pre{background:#f7f7f7;padding:14px;border-radius:6px;overflow-x:auto;font-size:13px}
pre code{background:none;padding:0}
img{max-width:100%;max-height:255mm;display:block;margin:14px 0;border:1px solid #e6e6e6;border-radius:4px}
blockquote{border-left:4px solid #d5d5d5;margin:16px 0;padding:6px 16px;color:#555;background:#fbfbfb}
@page{size:A4;margin:14mm 11mm}
h1,h2,h3{page-break-after:avoid;break-after:avoid}
tr,img,pre{page-break-inside:avoid;break-inside:avoid}
@media print{body{margin:0;padding:0;max-width:none;font-size:13px}
  h1{font-size:24px}h2{font-size:19px}h3{font-size:15px}table{font-size:11px}}
"""


def main(path: str) -> None:
    src = Path(path)
    html_body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    out = src.with_suffix(".html")
    out.write_text(
        f'<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{src.stem}</title><style>{CSS}</style></head>"
        f"<body>{html_body}</body></html>\n",
        encoding="utf-8",
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
