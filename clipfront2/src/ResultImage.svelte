<style lang="sass">
    .ch
        height: 100%
        img, video, picture, a
            height: 100%

    .cw
        width: 100%
        img, video, picture, a
            width: 100%

    *
        display: block
</style>

<div style={aspectRatio(result)} class={constrainBy === "width" ? " cw" : "ch"}>
    <a href={util.getURL(result)} on:click={() => interact("click")} on:mousedown={() => interact("mousedown")}>
        {#if util.hasFormat(results, result, "VIDEO")}
            <video controls poster={util.hasFormat(results, result, "jpegh") ? util.thumbnailURL(results, result, "jpegh") : null} preload="metadata" on:loadstart={updateCounter} on:loadedmetadata={redrawGrid} on:loadeddata={redrawGrid}>
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

<script>
    import * as util from "./util"

    export let result
    export let results
    export let updateCounter
    export let redrawGrid
    export let constrainBy = "width"

    const interact = type => {
        util.sendTelemetry("interact", {
            type,
            result: result[1]
        })
    }

    const aspectRatio = result => result[4] ? `aspect-ratio: ${result[4][0]}/${result[4][1]}` : null
</script>
