import * as config from "../../frontend_config.json"
import * as formats from "../../formats.json"

export const getURL = x => config.image_path + x

export const doQuery = args => fetch(config.backend_url, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(args)
}).then(x => x.json())

const filesafeCharset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
export const thumbnailPath = (originalPath, format) => {
    const extension = formats.formats[format][0]
    // Python and JS have minor differences in string handling wrt. astral characters which could result in incorrect quantities of dashes. Fortunately, Array.from handles this correctly.
    return config.thumb_path + `${Array.from(originalPath).map(x => filesafeCharset.includes(x) ? x : "_").join("")}.${format}${extension}`
}

const thumbedExtensions = formats.extensions
export const hasThumbnails = t => {
    const parts = t.split(".")
    return thumbedExtensions.includes("." + parts[parts.length - 1])
}