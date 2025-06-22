```mermaid
graph TD
    subgraph "Experiment Setup"
        A["Selection:
        4 Songs (Diverse Genres/Languages)
        2 Excerpts per Song (approx. 22s each)
        Total: 8 Excerpts"]
    end

    subgraph "Human Evaluation"
        B["13 Human Judges
        (Demographics: M_age=45.9, 69% F, high academic/music background)"]
        C["Task: Rate each of the 8 Excerpts
        - Perceived Valence (1-7 Likert)
        - Perceived Arousal (1-7 Likert)
        (Separately for Audio & Lyrical Content)"]
        D["Inter-Rater Reliability:
        ICC Calculated on Human Ratings
        (Result: Excellent Agreement for Averages)"]
        B --> C;
        C --> D;
    end

    subgraph "AI Evaluation"
        E["Same 8 Excerpts Processed by
        `song_sent_scores` Toolkit"]
        F["Output:
        - Audio V/A Scores (-1 to +1)
        - Text V/A Scores (-1 to +1)"]
        E --> F;
    end
    
    subgraph "Comparative Analysis"
        G["Data Preparation:
        - Average Human Scores per Item (32 items total)
        - Rescale Human Averages (1-7 scale to -1 to +1 scale)"]
        H["Comparison:
        Rescaled Human Averages
        vs.
        AI Scores (`song_sent_scores`)"]
        I["Statistical Method:
        Pearson's Correlation (r)
        (Overall, Per Song, Per Dimension)"]
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
    classDef ai fill:#D4F9D8,stroke:#333,stroke-width:1px,color:#000;
    class E,F ai;
    classDef analysis fill:#FFFACD,stroke:#333,stroke-width:1px,color:#000; 
    class G,H,I analysis;
