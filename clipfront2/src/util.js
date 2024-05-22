import * as config from "../../frontend_config.json"
import { writable } from "svelte/store"

export const getURL = x => config.image_path + x[1]

export const doQuery = args => fetch(config.backend_url, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(args)
}).then(x => x.json())

export const hasFormat = (results, result, format) => {
    return (results.formats.indexOf(format) != -1) && ((result[3] & (1 << results.formats.indexOf(format))) != 0)
}

export const thumbnailURL = (results, result, format) => {
    return `${config.thumb_path}${result[2]}${format}.${results.extensions[format]}`    
}

export let serverConfig = writable({})
fetch(config.backend_url).then(x => x.json().then(x => {
    serverConfig.set(x)
    window.serverConfig = x
}))