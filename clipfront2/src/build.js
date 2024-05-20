const esbuild = require("esbuild")
const sveltePlugin = require("esbuild-svelte")
const path = require("path")
const { sass } = require("svelte-preprocess-sass")

esbuild
    .build({
        entryPoints: [path.join(__dirname, "app.js")],
        bundle: true,
        minify: false,
        outfile: path.join(__dirname, "../static/app.js"),
        plugins: [sveltePlugin({
            preprocess: {
                style: sass()
            }
        })],
        loader: {
            ".woff": "file",
            ".woff2": "file",
            ".ttf": "file"
        },
        logLevel: "info",
        watch: process.argv.join(" ").includes("watch")
    })
    .catch(() => process.exit(1))
