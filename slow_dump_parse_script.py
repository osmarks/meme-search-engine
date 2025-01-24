import umsgpack
import zstandard
import pyarrow as pa

data = []

with open("sample.zst", "rb") as f:
    decomp = zstandard.ZstdDecompressor()
    reader = decomp.stream_reader(f)
    count = 0
    while True:
        try:
            url, id, title, subreddit, author, timestamp, embedding = umsgpack.unpack(reader)
            embedding = bytes(embedding)
            data.append({"url": url, "id": id, "title": title, "subreddit": subreddit, "author": author, "timestamp": timestamp, "embedding": embedding})
            count += 1
        except umsgpack.InsufficientDataException:
            break
    print(count)

schema = pa.schema([
    ("url", pa.string()),
    ("id", pa.string()),
    ("title", pa.string()),
    ("subreddit", pa.string()),
    ("author", pa.string()),
    ("timestamp", pa.int64()),
    ("embedding", pa.binary())
])

table = pa.Table.from_pylist(data, schema=schema)

with pa.OSFile("output.parquet", "wb") as sink:
    with pa.RecordBatchFileWriter(sink, schema) as writer:
        writer.write_table(table)