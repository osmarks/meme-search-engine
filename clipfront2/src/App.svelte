<style lang="sass">
    \:global(*)
        box-sizing: border-box

    \:global(html)
        scrollbar-color: black lightgray
            
    \:global(body)
        font-family: "Fira Sans", "Noto Sans", "Segoe UI", Verdana, sans-serif
        font-weight: 300
        overflow-anchor: none
        //margin: 0
        //min-height: 100vh

    \:global(strong)
        font-weight: bold

    @mixin header
        border-bottom: 1px solid gray
        margin: 0
        margin-bottom: 0.5em
        font-weight: 500
        //a
            //color: inherit

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

    input, button, select
        border-radius: 0
        border: 1px solid gray
        padding: 0.5em

    .controls
        input[type=search]
            width: 80%
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

    .result
        border: 1px solid gray
        *
            display: block
    .result img, .result video
        width: 100%
</style>

<h1>Meme Search Engine</h1>
{#if config.n_total}
    <p>{config.n_total} items indexed.</p>
{/if}
<details>
    <summary>Usage tips</summary>
    <ul>
        <li>This uses CLIP-like image/text embedding models. In general, search by thinking of what caption your desired image might be given by random people on the internet.</li>
        <li>The model can read text, but not all of it.</li>
        <li>In certain circumstances, it may be useful to postfix your query with "meme".</li>
        <li>Capitalization is ignored.</li>
        <li>Only English is supported. Other languages might work slightly.</li>
        <li>Sliders are generated from PCA on the index. The human-readable labels are approximate.</li>
        <li>Want your own deployment? Use the open-source code on <a href="https://github.com/osmarks/meme-search-engine/">GitHub.</a>.</li>
    </ul>
</details>
<div class="controls">
    <ul>
        {#each queryTerms as term}
            <li>
                <button on:click={removeTerm(term)}>Remove</button>
                <select bind:value={term.sign}>
                    <option>+</option>
                    <option>-</option>
                </select>
                <input type="range" min="0" max="2" bind:value={term.weight} step="0.01">
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

{#if error}
    <div>{error}</div>
{/if}
{#if resultPromise}
    <Loading />
{/if}
{#if results}
    {#if displayedResults.length === 0}
        No results. Wait for index rebuild.
    {/if}
    <Masonry bind:refreshLayout={refreshLayout} colWidth="minmax(Min(20em, 100%), 1fr)" items={displayedResults}>
        {#each displayedResults as result}
            {#key `${queryCounter}${result.file}`}
                <div class="result">
                    <a href={util.getURL(result)}>
                        {#if util.hasFormat(results, result, "VIDEO")}
                            <video controls poster={util.thumbnailURL(results, result, "jpegh")} preload="metadata" on:loadstart={updateCounter} on:loadedmetadata={redrawGrid}>
                                <source src={util.getURL(result)} />
                            </video>
                        {:else}
                            <picture>
                                {#if util.hasFormat(results, result, "avifl")}
                                    <source srcset={util.thumbnailURL(results, result, "avifl") + (util.hasFormat(results, result, "avifh") ? ", " + util.thumbnailURL(results, result, "avifh") + " 2x" : "")} type="image/avif" />
                                {/if}
                                {#if util.hasFormat(results, result, "jpegl")}
                                    <source srcset={util.thumbnailURL(results, result, "jpegl") + (util.hasFormat(results, result, "jpegh") ? ", " + util.thumbnailURL(results, result, "jpegh") + " 2x" : "")} type="image/jpeg" />
                                {/if}
                                <img src={util.getURL(result)} on:load={updateCounter} on:error={updateCounter} alt={result[1]}>
                            </picture>
                        {/if}
                    </a>
                </div>
            {/key}
        {/each}
    </Masonry>
{/if}

<svelte:window on:resize={redrawGrid} on:scroll={handleScroll}></svelte:window>

<script>
    import * as util from "./util"
    import Loading from "./Loading.svelte"
    import Masonry from "./Masonry.svelte"

    const chunkSize = 40

    let queryTerms = []
    let queryCounter = 0

    let config = {}
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
                weight: 0.2
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

    const focusEl = el => el.focus()
    const newTextQuery = (content=null) => {
        queryTerms.push({ type: "text", weight: 1, sign: "+", text: typeof content === "string" ? content : "" })
        queryTerms = queryTerms
    }
    const removeTerm = term => {
        queryTerms = queryTerms.filter(x => x !== term)
    }

    let refreshLayout
    let heightThreshold
    let error
    let pendingImageLoads
    const recomputeScroll = () => {
        const maxOffsets = new Map()
        for (const el of document.querySelectorAll(".result")) {
            if (el.getAttribute("data-h")) { // layouted
                const rect = el.getBoundingClientRect()
                maxOffsets.set(rect.left, Math.max(maxOffsets.get(rect.left) || 0, rect.top))
            }
        }
        heightThreshold = Math.min(...maxOffsets.values())
        console.log(heightThreshold, pendingImageLoads)
    }
    const redrawGrid = async () => {
        if (refreshLayout) {
            refreshLayout()
            await recomputeScroll()
        }
    }
    let resultPromise
    let results
    let displayedResults = []
    const runSearch = async () => {
        if (!resultPromise) {
            let args = {
                "terms": queryTerms.filter(x => x.text !== "").map(x => ({ image: x.imageData, text: x.text, embedding: x.embedding, predefined_embedding: x.predefinedEmbedding, weight: x.weight * { "+": 1, "-": -1 }[x.sign] })),
                "include_video": true
            }
            queryCounter += 1
            resultPromise = util.doQuery(args).then(res => {
                error = null
                results = res
                resultPromise = null
                displayedResults = []
                pendingImageLoads = 0
                for (let i = 0; i < chunkSize; i++) {
                    if (i >= results.matches.length) break
                    displayedResults.push(results.matches[i])
                    pendingImageLoads += 1
                }
                redrawGrid()
            }).catch(e => { error = e; resultPromise = null })
        }
    }

    const handleScroll = () => {
        if (window.scrollY + window.innerHeight >= heightThreshold && pendingImageLoads === 0) {
            recomputeScroll()
            if (window.scrollY + window.innerHeight < heightThreshold) return;
            let init = displayedResults.length
            for (let i = 0; i < chunkSize; i++) {
                if (init + i >= results.matches.length) break
                displayedResults.push(results.matches[init + i])
                pendingImageLoads += 1
            }
            displayedResults = displayedResults
        }
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
            console.log(currentFile)
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

    const updateCounter = () => {
        console.log("redraw")
        pendingImageLoads -= 1
        redrawGrid()
    }

    const queryStringParams = new URLSearchParams(window.location.search)
    if (queryStringParams.get("q")) {
        newTextQuery(queryStringParams.get("q"))
        runSearch()
    }
    if (queryStringParams.get("e")) {
        const binaryData = atob(queryStringParams.get("e").replace(/\-/g, "+").replace(/_/g, "/"))
        const uint16s = new Uint16Array(new Uint8Array(binaryData.split('').map(c => c.charCodeAt(0))).buffer)
        queryTerms.push({ type: "embedding", weight: 1, sign: "+", embedding: Array.from(uint16s).map(decodeFloat16) })
        runSearch()
    }
</script>
