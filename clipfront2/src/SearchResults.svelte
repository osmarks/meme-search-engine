<style lang="sass">
    @use 'common' as *

    .result
        border: 1px solid $palette-primary
        overflow: hidden

    .result img, .result video
        width: 100%

    .advanced
        margin-top: 1em

    .friendly
        padding-top: 1em
</style>

{#if error}
    <div class="error">{error}</div>
{/if}
{#if resultPromise}
    <Loading />
{/if}
{#if results}
<div class={friendlyMode ? "friendly" : "advanced"}>
    <Masonry bind:refreshLayout={refreshLayout} colWidth={`minmax(Min(${friendlyMode ? "30em" : "20em"}, 100%), 1fr)`} items={displayedResults} gridGap={friendlyMode ? "1em" : "0.5em"}>
        {#each displayedResults as result}
            {#key `${queryCounter}${result.file}`}
                <div class="result">
                    <ResultImage {result} {results} {updateCounter} {redrawGrid} constrainBy="width" />
                </div>
            {/key}
        {/each}
    </Masonry>
</div>
{/if}

<svelte:window on:resize={redrawGrid} on:scroll={handleScroll}></svelte:window>

<script>
    import { tick } from "svelte"

    import Loading from "./Loading.svelte"
    import Masonry from "./Masonry.svelte"
    import ResultImage from "./ResultImage.svelte"
    import * as util from "./util.js"

    let refreshLayout

    export let results
    export let resultPromise
    export let error
    export let friendlyMode
    export let queryCounter

    const chunkSize = 40

    let displayedResults = []

    let heightThreshold
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
    }
    export const redrawGrid = async () => {
        if (refreshLayout) {
            refreshLayout()
            await recomputeScroll()
        }
    }

    const updateCounter = () => {
        pendingImageLoads -= 1
        redrawGrid()
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
            if (init !== displayedResults.length) {
                util.sendTelemetry("scroll", {
                    results: results.matches.length,
                    displayed: displayedResults.length
                })
                displayedResults = displayedResults
            }
        }
    }

    let lastResults

    $: {
        if (results && results !== lastResults) {
            displayedResults = []
            pendingImageLoads = 0
            for (let i = 0; i < chunkSize; i++) {
                if (i >= results.matches.length) break
                displayedResults.push(results.matches[i])
                pendingImageLoads += 1
            }
            redrawGrid()
            lastResults = results
        }
    }

    $: {
        let _ = friendlyMode
        tick().then(() => redrawGrid())
    }
</script>
