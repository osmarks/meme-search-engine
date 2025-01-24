<style lang="sass">
    @use 'common' as *

    .about
        max-width: 40em
        padding-left: 1em
        padding-right: 1em
</style>

<div class="about">
    <p>
        Welcome to {util.hardConfig.name} by <a href="https://osmarks.net/">osmarks.net</a> Computational Memetics. {util.hardConfig.name} searches images using semantic image/text embedding models (refer to <a href="https://arxiv.org/abs/2303.15343">https://arxiv.org/abs/2303.15343</a>, <a href="https://arxiv.org/abs/2103.00020">https://arxiv.org/abs/2103.00020</a>). In general, search by thinking of what caption your desired image might have been given by random people on the internet. The model currently in use can read text fairly well and understands moderately abstract properties of images, but is limited to English and case-insensitive.
    </p>
    <p>
        "Useful"/"aesthetic"/"meme" sliders are defined based on an approximate extrapolation of my preferences and may not agree with your own opinions. Results are otherwise decided by the inscrutable whims of a billion-parameter neural network. Please note that large-scale internet data may contain things you do not like.
    </p>
    <p>
        The code is open-source and available on <a href="https://github.com/osmarks/meme-search-engine/">GitHub</a>.
    </p>
    {#if util.hardConfig.telemetry_endpoint}
        <h2>Privacy</h2>
        <p>
            We do not collect personal information. We do collect usage information (associated with a random ID) to improve the ranking algorithms. You can disable this:
        </p>
        <div>
            <input type="checkbox" checked={$telemetryStore} on:change={saveTelemetryEnabled} id="telemetry" />
            <label for="telemetry">Allow usage statistics</label>
        </div>
    {/if}
</div>

<script>
    import * as util from "./util"

    const telemetryStore = util.telemetryEnabled

    const saveTelemetryEnabled = ev => {
        telemetryStore.set(ev.target.checked)
    }
</script>
