Mastodon Misinformation Pipeline — Proof of Concept voor Bachelorproef

Deze repository bevat een machine learning pipeline voor het detecteren en classificeren van desinformatie op het Mastodon-netwerk. Het project omvat data-preprocessing, feature-engineering, modeltraining (inclusief klassieke modellen en BERT), evaluatie en visualisaties.

Inhoud

- Projectbeschrijving
- Projectstructuur
- Installatie en setup
- Gebruik
- Modellen en resultaten
- Testing
- Licentie en contact

Projectbeschrijving

Sociale netwerken zoals Mastodon zijn kwetsbaar voor de verspreiding van desinformatie. In deze bachelorproef is een pipeline ontwikkeld om Mastodon-berichten te analyseren en te classificeren.

Functionaliteiten

- Preprocessing: schoonmaken van Mastodon-specifieke tekst (HTML-tags, emoticons, hashtags, verwijzingen naar gebruikers).
- Modellen: klassieke baselines zoals Logistic Regression (LR) en Support Vector Machines (SVM).
- Deep learning: BERT (Bidirectional Encoder Representations from Transformers) voor geavanceerde NLP-classificatie.
- Evaluatie: berekening van metrics zoals F1-score, precision, recall en het genereren van verwarringsmatrices.

Projectstructuur

De repository is als volgt georganiseerd:

```text
.
├── data/
│   ├── mastodon/
│   ├── raw/
│   └── processed/
├── notebooks/
├── results/
│   ├── bert/
│   ├── figures/
│   ├── lr/
│   └── svm/
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── evaluation/
├── tests/
├── requirements.txt
├── run_pipeline.py
└── setup.py
```

Opmerking: Grote datasets (map `data/`) en getrainde deep learning-gewichten (bijvoorbeeld `results/bert/best_model/`) zijn uitgesloten van de Git-repository via `.gitignore`.

Installatie en setup

Vereisten

- Python 3.10 of hoger
- Optioneel: CUDA-compatibele GPU voor het trainen van BERT

Virtuele omgeving (aanbevolen)

Navigeer naar de projectmap en maak een virtuele omgeving aan:

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate
```

Dependencies installeren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Gebruik

Volledige pipeline

Start de volledige workflow (preprocessing → training → evaluatie) met:

```bash
python run_pipeline.py
```

Exploratieve data-analyse (EDA)

In de map `notebooks/` staan Jupyter Notebooks voor interactieve verkenning. Start Jupyter Lab of Notebook met:

```bash
jupyter notebook
```

Modellen en resultaatopslag

De pipeline traint en vergelijkt meerdere modelarchitecturen:

- Logistic Regression (`src/models/lr`): lichte baseline met TF-IDF features.
- Support Vector Machine (`src/models/svm`): geschikt voor hoge-dimensionale tekstrepresentaties.
- BERT (`src/models/bert`): transformer-gebaseerd model voor betere semantische representatie.

Alle trainingsrapporten, verwarringsmatrices en visualisaties worden opgeslagen in de map `results/`.

Testing

Unit tests bevinden zich in de map `tests/`. Voer ze uit met:

```bash
pytest tests/
```



Auteur: Kyanu Neckebroek

Instelling: Hogeschool Gent (HOGENT)

Academiejaar: 2025-2026

Opleiding: Bachelor in de Toegepaste Informatica


