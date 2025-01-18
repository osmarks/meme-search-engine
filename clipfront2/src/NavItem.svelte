<style lang="sass">
    @use 'common' as *

    a
        text-decoration: none
        font-size: 1.5em
        display: inline-block
        padding-right: 0.25em
        padding-left: 0.25em
        color: white

    .active
        font-style: italic
</style>

<a href={util.router.urlForPage(page)} class={$currentPage === page ? "active" : ""} on:click={util.router.handleClick}><slot /></a>

<script>
    import * as util from "./util"
    import { derived } from "svelte/store"

    let currentPage = derived(util.router, x => x.get("page"))

    export let page
</script>
