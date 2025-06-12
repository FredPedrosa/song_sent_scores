```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "primaryTextColor": "#000000",
    "lineColor": "#333333",
    "fontSize": "16px"
  }
}}%%

graph TD
    A["Input Audio
(e.g., song.mp3)"] --> B["CLAP Model
(Audio to V/A Scores)"];
    B --> C["Audio V/A Scores"];

    D["Optional:
Provided Lyrics"];
    E["ASR Model
(e.g., Whisper)"];
    F["Transcribed Text"];
    G["NLI Sentiment Model
(Text to V/A Scores)"];
    H["Text V/A Scores"];

    D -- "If lyrics provided" --> G;
    A -- "If no lyrics, for ASR" --> E;
    E --> F;
    F -- "If ASR used" --> G;
    
    G --> H;

    C --> I["Output Dictionary:
{'audio': {valence, arousal},
 'text':  {valence, arousal}}"];
    H --> I;
    
    I --> J["Visualization:
Circumplex Plot
(2D space: valence x arousal)
- Audio point
- Text point"];

    %% Aplicando classes aos nós
    class A,D,E,F,I,J default;
    class B,G model;
    class C,H scores;

    %% Definições de classe para cores (versão para tema claro)
    classDef default fill:#DDEFFD,stroke:#333333,stroke-width:1px,color:#000000;
    classDef model fill:#C9D4F9,stroke:#333333,stroke-width:1px,color:#000000;
    classDef scores fill:#D4F9D8,stroke:#333333,stroke-width:1px,color:#000000;
