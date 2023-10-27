import sqlite3
import os
import hashlib
import json
import string
import subprocess
from PIL import Image
import tempfile
import shutil
import math
import sys

with open(sys.argv[1], "r") as config_file:
    CONFIG = json.load(config_file)

filesafe_charset = string.ascii_letters + string.digits + "-"
def avif_format(quality):
    avif_speed = "4"
    def fn(inpath, outpath):
        if os.path.splitext(inpath)[-1].lower() not in {".jpg", ".png", ".jpeg", ".avif"}:
            with tempfile.NamedTemporaryFile() as tf:
                subprocess.run(["convert", inpath, "png:" + tf.name])
                subprocess.run(["avifenc", "-s", avif_speed, "-j", "all", "-q", str(quality), tf.name, outpath], capture_output=True).check_returncode()
        else:
            subprocess.run(["avifenc", "-s", avif_speed, "-j", "all", "-q", str(quality), inpath, outpath], capture_output=True).check_returncode()
    return fn

def jpeg_format(quality=None, maxwidth=None, maxheight=None, target_size=None):
    def do_convert(size, quality, input, output):
        subprocess.run(["convert", input, "-resize", "x".join(map(str, size)), "-quality", str(quality), output]).check_returncode()
    def fn(inpath, outpath):
        im = Image.open(inpath)
        width, height = im.size
        if maxwidth and width > maxwidth:
            height /= width / maxwidth
            height = math.floor(height)
            width = maxwidth
        if maxheight and height > maxheight:
            width /= height / maxheight
            width = math.floor(width)
            height = maxheight
        if target_size is None:
            do_convert((width, height), quality, inpath, outpath)
        else:
            q_min = 1
            q_max = 100
            while True:
                with tempfile.NamedTemporaryFile() as tf:
                    test_quality = (q_min + q_max) // 2
                    do_convert((width, height), test_quality, inpath, tf.name)
                    stat = os.stat(tf.name)
                    if stat.st_size >= target_size:
                        # too big
                        q_max = test_quality
                    else:
                        q_min = test_quality + 1
                    if q_min >= q_max:
                        shutil.copy(tf.name, outpath)
                        break
        
    return fn

input_path = CONFIG["input"]
output_path = CONFIG["output"]
exts = {".webp", ".png", ".jpg", ".jpeg"}
output_formats = {
    "avif-lq": (avif_format(quality=30), ".avif", "image/avif"),
    "avif-hq": (avif_format(quality=80), ".avif", "image/avif"),
    "jpeg-800": (jpeg_format(maxwidth=800, quality=80), ".jpeg", "image/jpeg"),
    "jpeg-fullscale": (jpeg_format(quality=80), ".jpeg", "image/jpeg"),
    "jpeg-256k": (jpeg_format(target_size=256_000, maxwidth=600, maxheight=600), ".jpeg", "image/jpeg")
}

with open("formats.json", "w") as f:
    json.dump({
        "formats": { k: v[1:] for k, v in output_formats.items() },
        "extensions": list(exts)
    }, f)

if "gen-formats" in sys.argv: raise SystemExit

con = sqlite3.connect(CONFIG["database"])
con.executescript("""
CREATE TABLE IF NOT EXISTS thumb (
    file TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    formats BLOB NOT NULL
);
""")
con.row_factory = sqlite3.Row

out_formats_set = set(output_formats)
def generate_output_format_string(formats):
    return json.dumps(sorted(formats))
def to_outpath(input, format):
    format_ext = output_formats[format][1]
    return f"{''.join([ i if i in filesafe_charset else '_' for i in input ])}" + "." + format + format_ext
full_formats = generate_output_format_string(output_formats.keys())

for directory, subdirectories, files in os.walk(input_path):
    directory = os.path.join(input_path, directory)
    if directory.startswith(output_path): continue
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[-1].lower()
        if ext in exts:
            path = os.path.join(directory, file)
            rawname = path.removeprefix(input_path).removeprefix("/")
            st = os.stat(path)
            csr = con.execute("SELECT mtime, formats FROM thumb WHERE file = ?", (rawname,))
            row = csr.fetchone()
            if not row:
                mtime, formats = None, "[]"
            else:
                mtime, formats = row
            if st.st_mtime != mtime or formats != full_formats:
                formats = set(json.loads(formats))
                for new_format in out_formats_set - formats:
                    new_path = os.path.join(output_path, to_outpath(rawname, new_format))
                    try:
                        output_formats[new_format][0](path, new_path)
                    except:
                        print("working on", new_format, rawname)
                        raise
                    nst = os.stat(new_path)
                    if nst.st_size > st.st_size: # bigger, so redundant
                        os.unlink(new_path)
                        os.symlink(os.path.relpath(path, output_path), new_path)
                    formats.add(new_format)
                con.execute("INSERT OR REPLACE INTO thumb VALUES (?, ?, ?)", (rawname, st.st_mtime, generate_output_format_string(formats)))
                con.commit()
                sys.stdout.write(".")
                sys.stdout.flush()