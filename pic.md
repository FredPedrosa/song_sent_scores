```mermaid
flowchart TD
    A["Input Audio
(e.g., song.mp3)"] --> B["CLAP Model
(Audio to V/A Scores)"]
    B --> C["Audio V/A Scores"]
    D["Optional:
Provided Lyrics"] -- If lyrics provided --> G["NLI Sentiment Model
(Text to V/A Scores)"]
    A -- If no lyrics --> E["ASR Model
(e.g., Whisper)"]
    E --> F["Transcribed Text"]
    F -- If ASR used --> G
    G --> H["Text V/A Scores"]
    C --> I@{ label: "Output Dictionary:\n{'audio': {valence, arousal},\n 'text':  {valence, arousal}}" }
    H --> I
    I --> J["Visualization:
Circumplex Plot
(2D space: valence x arousal)
Audio point
Text point"];
    I@{ shape: rect}
     A:::default
     B:::model
     C:::scores
     D:::default
     G:::model
     E:::default
     F:::default
     H:::scores
     I:::default
     J:::default
    classDef default fill:#f9f,stroke:#333,stroke-width:1px
    classDef model fill:#bbf,stroke:#333,stroke-width:1px
    classDef scores fill:#dfd,stroke:#333,stroke-width:1px
