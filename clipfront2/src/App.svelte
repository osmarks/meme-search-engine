<style lang="sass">
    \:global(*)
        box-sizing: border-box

    \:global(html)
        scrollbar-color: black lightgray
            
    \:global(body)
        font-family: "Fira Sans", "Noto Sans", "Segoe UI", Verdana, sans-serif
        font-weight: 300
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

    .result
        border: 1px solid gray
    .result img
        width: 100%
</style>

<h1>Meme Search Engine</h1>
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
            </li>
        {/each}
    </ul>
    <div class="ctrlbar">
        <input type="search" placeholder="Text Query" on:keydown={handleKey} on:focus={newTextQuery}>
        <button on:click={pickFile}>Image Query</button>
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
    <Masonry bind:refreshLayout={refreshLayout} colWidth="minmax(Min(20em, 100%), 1fr)" items={displayedResults}>
        {#each displayedResults as result}
            {#key result.file}
                <div class="result"><a href={util.getURL(result.file)}><img src={util.getURL(result.file)} on:load={updateCounter} on:error={updateCounter} alt={result.caption || result.file}></a></div>
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

    const focusEl = el => el.focus()
    const newTextQuery = (content="") => {
        queryTerms.push({ type: "text", weight: 1, sign: "+", text: content })
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
    const redrawGrid = () => {
        if (refreshLayout) refreshLayout().then(recomputeScroll)
    }
    let resultPromise
    let results
    let displayedResults = []
    const runSearch = async () => {
        if (!resultPromise) {
            let args = {}
            args.text = queryTerms.filter(x => x.type === "text" && x.text).map(({ text, weight, sign }) => [ text, weight * { "+": 1, "-": -1 }[sign] ])
            args.images = queryTerms.filter(x => x.type === "image").map(({ imageData, weight, sign }) => [ imageData, weight * { "+": 1, "-": -1 }[sign] ])
            resultPromise = util.doQuery(args).then(res => {
                error = null
                results = res
                resultPromise = null
                displayedResults = []
                pendingImageLoads = 0
                for (let i = 0; i < chunkSize; i++) {
                    displayedResults.push(results[i])
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
                displayedResults.push(results[init + i])
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
</script>
