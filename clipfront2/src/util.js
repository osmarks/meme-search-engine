import * as config from "../../frontend_config.json"
import * as backendConfig from "../../mse_config.json"
import * as formats from "../../formats.json"

export const getURL = x => config.image_path + x[1]

export const doQuery = args => fetch(config.backend_url, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(args)
}).then(x => x.json())

export const hasFormat = (results, result, format) => {
    return result[3] && (1 << results.formats.indexOf(format)) !== 0
}

export const thumbnailURL = (results, result, format) => {
    console.log("RES", results)
    return `${config.thumb_path}${result[2]}${format}.${results.extensions[format]}`    
}