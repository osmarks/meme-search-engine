import * as config from "../../frontend_config.json"
import { writable, get } from "svelte/store"

export const getURL = x => config.image_path + x[1]

export const hardConfig = config

export const router = writable(new URLSearchParams(window.location.search))

window.addEventListener("popstate", ev => {
    router.set(new URLSearchParams(window.location.search))
})

router.handleClick = ev => {
    history.pushState({}, "", ev.target.getAttribute("href"))
    ev.preventDefault()
    router.set(new URLSearchParams(window.location.search))
}

router.urlForPage = page => {
    let queryStringParams = new URLSearchParams(window.location.search)
    queryStringParams.set("page", page)
    return window.location.origin + "?" + queryStringParams.toString()
}

export const telemetryEnabled = writable(true)
if (localStorage.telemetryEnabled === "false") {
    telemetryEnabled.set(false)
}
telemetryEnabled.subscribe(x => {
    localStorage.telemetryEnabled = x ? "true" : "false"
})

const randomString = () => Math.random().toString(36).substring(2, 15)

localStorage.correlationId = localStorage.correlationId ?? randomString()
let correlationId = localStorage.correlationId
let instanceId = randomString()

export const sendTelemetry = async (event, data) => {
    if (!get(telemetryEnabled)) return
    if (!config.telemetry_endpoint) return
    navigator.sendBeacon(config.telemetry_endpoint, JSON.stringify({
            correlationId,
            instanceId,
            event,
            data,
            page: get(router).get("page")
        })
    )
}

export const doQuery = async args => {
    const res = await fetch(config.backend_url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(args)
    })
    try {
        return await res.clone().json()
    } catch(e) {
        throw new Error(res.status + " " + await res.text())
    }
}

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
