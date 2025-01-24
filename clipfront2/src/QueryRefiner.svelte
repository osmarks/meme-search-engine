<style lang="sass">
    .candidate-images
        height: 15vh
        display: flex

    .candidate
        margin-top: 1em
        .candidate-images
            margin-top: 1em
</style>

<div>
    <div>
        {#each candidates as candidate}
            <div class="candidate">
                <button on:click={select(candidate)}>Select {candidate.i + 1}</button>
                <div class="candidate-images">
                    {#if candidate.results}
                        {#each candidate.results.matches as result}
                            <ResultImage {result} results={candidate.results} updateCounter={null} redrawGrid={null} constrainBy="height" />
                        {/each}
                    {/if}
                </div>
            </div>
        {/each}
    </div>
</div>

<svelte:window on:keydown={handleKey} />

<script>
    import * as util from "./util"
    import ResultImage from "./ResultImage.svelte"

    export let config

    const d_emb = 1152

    const K = 2
    let candidates = []

    const select = async candidate => {
        candidates = []
        const direction = util.randn(d_emb, 1 / d_emb)
        for (let i = -K; i <= K; i++) {
            const newV = util.vecSum(util.vecScale(direction, i / K), candidate.vector)
            candidates.push({ vector: newV, results: null, i: i + K })
        }
        await Promise.all(candidates.map(async x => {
            const queryResult = await util.doQuery({ terms: [{ embedding: x.vector, weight: 1, sign: "+" }], include_video: false, k: 100 })
            x.results = queryResult
            x.results.matches = x.results.matches.slice(0, 10)
        }))
        candidates = candidates
        console.log(candidates)
    }

    select({ vector: util.randn(d_emb, 1 / d_emb) })

    const handleKey = ev => {
        const num = parseInt(ev.key)
        if (num && num >= 1 && num <= (2 * K + 1)) {
            select(candidates[num - 1])
        }
    }
</script>
