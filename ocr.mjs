import Lens from 'chrome-lens-ocr'
import sharp from "sharp"
import fs from "fs/promises"
import sqlite3 from "better-sqlite3"
import path from "path"

import memeSearchConfig from "./mse_config.json" with { type: "json" }
import ocrConfig from "./ocr_config.json" with { type: "json" }

const DB = sqlite3(memeSearchConfig.db_path)

DB.exec(`
CREATE TABLE IF NOT EXISTS ocr (
    filename TEXT PRIMARY KEY REFERENCES files(filename),
    scan_time INTEGER NOT NULL,
    text TEXT NOT NULL,
    raw_segments TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5 (
    filename,
    text,
    tokenize='unicode61 remove_diacritics 2',
    content='ocr'
);

CREATE TRIGGER IF NOT EXISTS ocr_fts_ins AFTER INSERT ON ocr BEGIN
    INSERT INTO ocr_fts (rowid, filename, text) VALUES (new.rowid, new.filename, new.text);
END;

CREATE TRIGGER IF NOT EXISTS ocr_fts_del AFTER DELETE ON ocr BEGIN
    INSERT INTO ocr_fts (ocr_fts, rowid, filename, text) VALUES ('delete', old.rowid, old.filename, old.text);
END;
`)

const preparedStatements = new Map()
const SQL = (strings, ...params) => {
    const sql = strings.join("?")
    let stmt
    const cachedValue = preparedStatements.get(sql)
    if (!cachedValue) {
            stmt = DB.prepare(sql)
            preparedStatements.set(sql, stmt)
    } else {
            stmt = cachedValue
    }
    return {
            get: () => stmt.get.apply(stmt, params),
            run: () => stmt.run.apply(stmt, params),
            all: () => stmt.all.apply(stmt, params),
            statement: stmt
    }
}

const wait = timeout => new Promise(resolve => setTimeout(resolve, timeout))

const lens = new Lens(ocrConfig.lens_options || {})

for (const file of SQL`SELECT files.filename FROM files LEFT JOIN ocr ON files.filename = ocr.filename WHERE ocr.scan_time IS NULL OR ocr.scan_time < files.modtime`.all()) {
    console.log(file.filename)
    const filepath = path.join(memeSearchConfig.files, file.filename)
    const metadata = await sharp(filepath).metadata()
    console.log(metadata.width, metadata.height)
    let newWidth = Math.min(metadata.width, ocrConfig.image_dim)
    let newHeight = Math.ceil(metadata.height * (newWidth / metadata.width))
    let text = ""
    let segments = []
    let failed = false
    for (let y = 0; y < newHeight; y += ocrConfig.image_dim) {
        const result = await sharp(filepath).resize(newWidth, newHeight, { fit: "fill" }).extract({
            left: 0,
            width: newWidth,
            top: y,
            height: Math.min(ocrConfig.image_dim, newHeight - y)
        }).png().toBuffer()
        let chunk
        let count = 10
        while (!chunk) {
            try {
                chunk = await lens.scanByBuffer(result)
            } catch(e) {
                console.log("OCR failed, retry", e.body ? "?" : e, count)
                await wait(500)
                count--
                if (count === 0) {
                    console.log("retry limit")
                    failed = true
                    break
                }
            }
        }
        if (failed) break
        // they appear to be in the "right order" out of the API anyway
        for (const segment of chunk.segments) {
            text += segment.text + "\n"
            segments.push({
                text: segment.text,
                x: segment.boundingBox.pixelCoords.x,
                y: segment.boundingBox.pixelCoords.y + y,
                width: segment.boundingBox.pixelCoords.width,
                height: segment.boundingBox.pixelCoords.height
            })
        }
    }
    if (failed) continue
    SQL`INSERT OR REPLACE INTO ocr VALUES (${file.filename}, ${Date.now() / 1000}, ${text.trim()}, ${JSON.stringify(segments)})`.run()
}