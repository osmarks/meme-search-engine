import * as config from "../../frontend_config.json"

export const getURL = x => config.image_path + x

export const doQuery = args => fetch(config.backend_url, {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(args)
}).then(x => x.json())