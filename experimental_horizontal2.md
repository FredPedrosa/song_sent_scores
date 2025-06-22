```mermaid
graph LR
    subgraph "Experiment Setup"
        A["Selection:
        4 Songs (Diverse genres/languages)
        2 Excerpts per song (approx. 22s each)
        Total: 8 excerpts"]
    end

    subgraph "Human Evaluation"
        B["13 Human Judges
        (Demographics: M_age=45.9, 69% F, diverse academic/music background)"]
        C["Task: Rate each of the 8 excerpts
        - Perceived Valence 
        - Perceived Arousal 
        (Audio & Lyrical)"]
        D["Inter-Rater Reliability:
        ICC on human ratings
        (Result: excellent agreement for averages)"]
        B --> C;
        C --> D;
    end

    subgraph "AI Evaluation"
        E["Same 8 Excerpts Processed
        (song_sent_scores Toolkit)"]
        F["Output:
        - Audio V/A Scores 
        - Text V/A Scores "]
        E --> F;
    end
    
    subgraph "Comparative Analysis"
        G["Data Preparation:
        - Average Human Scores per Item (32 items total)
        - Rescale Human Averages "]
        H["Comparison:
        Rescaled Human Averages
        vs.
        AI Scores (song_sent_scores)"]
        I["Statistical Method:
        Pearson's Correlation (r)"]
        D --> G;
        F --> H;
        G --> H;
        H --> I;
    end
    
    A --> B;
    A --> E;

    classDef setup fill:#DDEFFD,stroke:#333,stroke-width:1px,color:#000; 
    class A setup;
    classDef human fill:#C9D4F9,stroke:#333,stroke-width:1px,color:#000;
    class B,C,D human;
    classDef ai fill:#D4F9D8,stroke:#333,stroke-width:1px,color:#000;
    class E,F ai;
    classDef analysis fill:#FFFACD,stroke:#333,stroke-width:1px,color:#000; 
    class G,H,I analysis;
