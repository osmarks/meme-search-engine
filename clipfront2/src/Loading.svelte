<style lang="sass">
    .spinner
        color: black
  
    .spinner:before
        animation: textSpinner 0.8s linear infinite
        content: "⠋"
        margin-right: 0.5em
        padding-top: 0.5em
    
    @keyframes textSpinner
        10% 
            content: "⠙"
        20% 
            content: "⠹"
        30% 
            content: "⠸"
        40% 
            content: "⠼"
        50% 
            content: "⠴"
        60% 
            content: "⠦"
        70% 
            content: "⠧"
        80% 
            content: "⠇"
        90% 
            content: "⠏"
</style>

<script>
    export let operation = "Loading"
</script>

<span class="spinner">{operation}</span>