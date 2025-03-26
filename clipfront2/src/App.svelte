<style lang="sass">
    @use 'common' as *
    @use 'sass:color'

    \:global(*)
        box-sizing: border-box

    \:global(html)
        scrollbar-color: black lightgray

    \:global(body)
        font-family: "Iosevka", "Shure Tech Mono", "IBM Plex Mono", monospace // TODO import iosevka
        font-weight: 300
        overflow-anchor: none
        margin: 0

    \:global(strong)
        font-weight: bold

    @mixin header
        border-bottom: 1px solid white
        margin: 0
        margin-bottom: 0.5em
        font-weight: 500

    h1
        text-transform: uppercase

    \:global(h1)
        @include header
    \:global(h2)
        @include header
    \:global(h3)
        @include header
    \:global(h4)
        @include header
    \:global(h5)
        @include header
    \:global(h6)
        @include header
    \:global(ul)
        list-style-type: square
        padding: 0
        padding-left: 1em

    \:global(input), :\global(button), :\global(select)
        border-radius: 0
        border: 1px solid gray
        padding: 0.5em
        font-family: inherit

    .controls
        input[type=search]
            width: 70%
        .ctrlbar
            > *
                margin: 0 -1px
            width: 100%
        ul
            list-style-type: none
            padding: 0
            li
                display: flex
                align-items: center
                > *
                    margin: 0 2px
        .sliders-ctrl
            width: 5em

    .enable-advanced-mode
        position: fixed
        top: 0.2em
        right: 0.2em
        font-size: 1.3em

    nav
        background: $palette-secondary
        display: flex
        justify-content: space-between
        padding: 1em

    .friendly
        h1
            border: none
            padding-top: 1em

        input[type=search]
            width: 100%
            margin: 0
            padding: 0.5em
            font-size: 1.5em
            border-radius: 6px

        .center
            margin-left: auto
            margin-right: auto
            max-width: 40em

        .description
            opacity: 0.8
            margin-bottom: 1em
            font-weight: bold

        button
            margin-left: 0.5em
            margin-right: 0.5em
            padding: 0.5em
            background: #9a0eea
            border-radius: 10px
            font-size: 1.5em
            color: white

        .header
            padding-bottom: 2em

    .header
        padding-left: 1em
        padding-right: 1em
        padding-top: 0.5em
        padding-bottom: 0.5em
        background: $palette-primary
        color: white

        p
            font-weight: bold

    .results
        padding-left: 1em
        padding-right: 1em

    @media (prefers-color-scheme: dark)
        \:global(body)
            background-color: black
            color: white

        \:global(input), :\global(button), :\global(select)
            border-radius: 0
            border: 1px solid gray
            padding: 0.5em
            font-family: inherit
            background: #222
            color: white

    .logo
        height: 1em
        vertical-align: middle
        margin-bottom: 6px
</style>

<nav>
    <div class="left">
        <NavItem page="search">Search</NavItem>
    </div>
    <div class="right">
        <NavItem page="advanced">Advanced</NavItem>
        <NavItem page="about">About</NavItem>
    </div>
</nav>

{#if page === "search" || page === "advanced"}
    <div class={"container" + (friendlyMode ? " friendly" : " advanced")}>
    <div class="header">
        {#if friendlyMode}
        <div>
            <div class="center">
                <h1 class="logo-container"><img class="logo" src="./logo.png"> {util.hardConfig.name}</h1>
                <div class="description">{util.hardConfig.description}</div>
                <input type="search" placeholder="🔎 Search Memes" on:keydown={handleKey} autofocus bind:value={friendlyModeQuery} />
            </div>
        </div>
        {:else}
        <h1>{util.hardConfig.name}</h1>
        <p>
        {#if config.n_total}
            {config.n_total} items indexed.
        {/if}
        </p>
        <div class="controls">
            <ul>
                {#each queryTerms as term}
                    <li>
                        <button on:click={removeTerm(term)}>Remove</button>
                        <select bind:value={term.sign}>
                            <option>+</option>
                            <option>-</option>
                        </select>
                        {#if debugEnabled}
                            <input type="number" bind:value={term.weight} step="0.01">
                        {:else}
                            <input type="range" min="0" max="2" bind:value={term.weight} step="0.01">
                        {/if}
                        {#if term.type === "image"}
                            <span>{term.file.name}</span>
                        {:else if term.type === "text"}
                            <input type="search" use:focusEl on:keydown={handleKey} bind:value={term.text} />
                        {/if}
                        {#if term.type === "embedding"}
                            <span>[embedding loaded from URL]</span>
                        {/if}
                        {#if term.type === "predefined_embedding"}
                            <span>{term.sign === "-" ? invertEmbeddingDesc(term.predefinedEmbedding) : term.predefinedEmbedding}</span>
                        {/if}
                    </li>
                {/each}
            </ul>
            {#if showDebugSwitch}
            <div class="ctrlbar">
                <input type="checkbox" bind:checked={debugEnabled} id="debug" />
                <label for="debug">Debug</label>
                {#if debugEnabled}
                {/if}
            </div>
            {/if}
            <div class="ctrlbar">
                <input type="search" placeholder="Text Query" on:keydown={handleKey} on:focus={newTextQuery}>
                <button on:click={pickFile}>Image Query</button>
                <select bind:value={predefinedEmbeddingName} on:change={setPredefinedEmbedding} class="sliders-ctrl">
                    <option>Sliders</option>
                    {#each config.predefined_embedding_names ?? [] as name}
                        <option>{name}</option>
                    {/each}
                </select>
                <button on:click={runSearch} style="margin-left: auto">Search</button>
            </div>
        </div>
        {/if}
    </div>

    <div class="results"><SearchResults {resultPromise} {results} {error} {friendlyMode} {queryCounter} /></div>
    </div>
{/if}

{#if page === "about"}
    <About />
{/if}

{#if page === "refine"}
    <QueryRefiner {config} />
{/if}

<script>
    import * as util from "./util"
    import SearchResults from "./SearchResults.svelte"
    import QueryRefiner from "./QueryRefiner.svelte"
    import NavItem from "./NavItem.svelte"
    import About from "./About.svelte"

    document.title = util.hardConfig.name

    let page = "search"
    let friendlyModeQuery = ""
    let queryTerms = []
    let queryCounter = 0
    let config = {}

    const showDebugSwitch = localStorage.getItem("debugEnabled") === "true"
    let debugEnabled = false

    const newTextQuery = (content=null) => {
        queryTerms.push({ type: "text", weight: 1, sign: "+", text: typeof content === "string" ? content : "" })
        queryTerms = queryTerms
    }

    let resultPromise
    let results
    let error

    const runSearch = async () => {
        if (!resultPromise) {
            const friendlyModeQueryOrRandom = friendlyModeQuery ? [{ text: friendlyModeQuery, weight: 1, sign: "+" }] : [{ embedding: util.randn(config.d_emb, 1 / (config.d_emb ** 0.5)), weight: 1, sign: "+" }]
            const terms = friendlyMode ?
                friendlyModeQueryOrRandom.concat(util.hardConfig.friendly_mode_default_terms ?? []) :
                queryTerms.filter(x => x.text !== "").map(x => ({ image: x.imageData, text: x.text, embedding: x.embedding, predefined_embedding: x.predefinedEmbedding, weight: x.weight * { "+": 1, "-": -1 }[x.sign] }))
            let args = {
                "terms": terms,
                "include_video": true,
                "debug_enabled": debugEnabled
            }

            util.sendTelemetry("search", {
                terms: args.terms.map(x => {
                    if (x.image) {
                        return { type: "image" }
                    } else if (x.text) {
                        return { type: "text", text: x.text }
                    } else if (x.embedding) {
                        return { type: "embedding" }
                    } else if (x.predefined_embedding) {
                        return { type: "predefined_embedding", embedding: x.predefined_embedding }
                    }
                })
            })

            queryCounter += 1
            resultPromise = util.doQuery(args).then(res => {
                error = null
                results = res
                resultPromise = null
            }).catch(e => { error = e; resultPromise = null; console.log("error", e) })
        }
    }

    const decodeFloat16 = uint16 => {
        const sign = (uint16 & 0x8000) ? -1 : 1
        const exponent = (uint16 & 0x7C00) >> 10
        const fraction = uint16 & 0x03FF

        if (exponent === 0) {
            return sign * Math.pow(2, -14) * (fraction / Math.pow(2, 10))
        } else if (exponent === 0x1F) {
            return fraction ? NaN : sign * Infinity
        } else {
            return sign * Math.pow(2, exponent - 15) * (1 + fraction / Math.pow(2, 10))
        }
    }

    const parseQueryString = queryStringParams => {
        if (queryStringParams.get("q") && queryTerms.length === 0) {
            newTextQuery(queryStringParams.get("q"))
            friendlyModeQuery = queryStringParams.get("q")
            runSearch()
        }
        if (queryStringParams.get("e") && queryTerms.length === 0) {
            const binaryData = atob(queryStringParams.get("e").replace(/\-/g, "+").replace(/_/g, "/"))
            const uint16s = new Uint16Array(new Uint8Array(binaryData.split('').map(c => c.charCodeAt(0))).buffer)
            queryTerms.push({ type: "embedding", weight: 1, sign: "+", embedding: Array.from(uint16s).map(decodeFloat16) })
            friendlyMode = false
            runSearch()
        }
        if (queryStringParams.get("page")) {
            page = queryStringParams.get("page")
        }
    }

    util.router.subscribe(parseQueryString)

    $: friendlyMode = page === "search"

    util.serverConfig.subscribe(x => {
        config = x
    })
    let predefinedEmbeddingName = "Sliders"

    const setPredefinedEmbedding = () => {
        if (predefinedEmbeddingName !== "Sliders") {
            queryTerms.push({
                type: "predefined_embedding",
                predefinedEmbedding: predefinedEmbeddingName,
                sign: "+",
                weight: 1
            })
        }
        queryTerms = queryTerms
        predefinedEmbeddingName = "Sliders"
    }

    const invertEmbeddingDesc = x => {
        const [fst, snd] = x.split("/")
        if (snd === undefined) return "Not " + fst
        return `${snd}/${fst}`
    }

    const focusEl = el => el.focus()
    const removeTerm = term => {
        queryTerms = queryTerms.filter(x => x !== term)
    }

    const handleKey = ev => {
        if (ev.key === "Enter") {
            runSearch()
        }
    }

    const input = document.createElement("input")
    input.type = "file"
    const pickFile = () => {
        input.oninput = ev => {
            currentFile = ev.target.files[0]
            if (currentFile) {
                let reader = new FileReader()
                reader.readAsDataURL(currentFile)
                let term = { type: "image", file: currentFile, weight: 1, sign: "+" }
                queryTerms.push(term)
                queryTerms = queryTerms
                reader.onload = () => {
                    term.imageData = reader.result.split(',')[1]
                }
            }
        }
        input.click()
    }
</script>
