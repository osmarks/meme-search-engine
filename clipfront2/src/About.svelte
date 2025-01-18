<style lang="sass">
    @use 'common' as *

    .about
        max-width: 40em
        padding-left: 1em
        padding-right: 1em
</style>

<div class="about">
    <p>
        Welcome to {util.hardConfig.name} by <a href="https://osmarks.net/">osmarks.net Computational Memetics</a>. {util.hardConfig.name} searches images using semantic image/text embedding models. In general, search by thinking of what caption your desired image might have been given by random people on the internet. The model currently in use can read text fairly well and understands moderately abstract properties of images, but is limited to English and case-insensitive.
    </p>
    <p>
        Advanced Mode sliders are generated from PCA on the index. The human-readable labels are generated manually by <a href="https://datasets.osmarks.net/components.html">looking at things</a>.
    </p>
    <p>
        The code is open-source and available on <a href="https://github.com/osmarks/meme-search-engine/">GitHub.</a>
    </p>
    {#if util.hardConfig.telemetryEndpoint}
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
