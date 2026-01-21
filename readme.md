# Multilingual Named Entity Recognition with mBERT

## Project Overview

This project implements a **two-stage approach** to Named Entity Recognition using **mBERT** (Multilingual BERT) as part of Module 4 (Transformers) for the CAS NLP 2025 program.

**Stage 1 - English Baseline:**
- Dataset: CoNLL-2003 (gold standard)
- Purpose: Compare BiLSTM (Module 3) vs mBERT (Module 4) architectures
- Result: 91.46% F1 (significant improvement over BiLSTM's 84.5%)

**Stage 2 - Multilingual Extension:**
- Languages: German, French, Dutch
- Dataset: WikiANN (5K sentences per language per split)
- Purpose: Demonstrate cross-lingual transfer learning capabilities
- Result: 88.93% F1 across three languages

**Rationale for two-stage approach:**
- CoNLL-2003 German unavailable due to copyright restrictions
- WikiANN provides uniform annotations for fair cross-lingual comparison
- English baseline uses gold standard for accurate architecture comparison
- Multilingual stage demonstrates mBERT's cross-lingual capabilities

**Key components:**
- Model: `bert-base-multilingual-cased` (177M parameters)
- Entity types: PER (person), LOC (location), ORG (organization), MISC (CoNLL only)
- Evaluation: Entity-level F1 using seqeval

## Project Structure
```
MNER/
├── data/
│   ├── raw/
│   │   ├── conll2003/          # English (CoNLL-2003 gold standard)
│   │   └── wikiann/            # Multilingual (WikiANN silver standard, 5K/split)
│   │       ├── de/
│   │       ├── fr/
│   │       └── nl/
│   └── raw_full/               # Original full WikiANN datasets (archived)
├── notebooks/
│   ├── M4_MNER.ipynb                  # Stage 1: English baseline
│   └── M4_MNER_multilingual.ipynb     # Stage 2: Multilingual (DE/FR/NL)
├── .gitignore
├── readme.md
└── requirements.txt

Note: models/ and data/processed/ excluded from repository (see .gitignore)
```

## Dataset

### Stage 1 - English (CoNLL-2003)

**Dataset characteristics:**
- Gold standard annotations (manual, high quality)
- 4 entity types: PER, LOC, ORG, MISC
- IOB2 tagging format (after in-memory correction of IOB1 PER entities)

**Dataset size:**
- Train: 14,041 sentences
- Dev: 3,250 sentences  
- Test: 3,453 sentences
- Total: 20,744 sentences

### Stage 2 - Multilingual (WikiANN)

**Dataset characteristics:**
- Silver standard annotations (Wikipedia-based, semi-automatic)
- 3 entity types: PER, LOC, ORG (no MISC)
- Consistent annotation scheme across languages
- IOB2 tagging format

**Dataset size (per language):**
- Train: 5,000 sentences
- Dev: 5,000 sentences
- Test: 5,000 sentences
- **Combined (DE+FR+NL):** 15K train / 15K dev / 15K test (45K total)
- Reduced from original (~20K/10K/10K per language) for practical CPU training time

**Rationale for WikiANN:**
- CoNLL-2003 German unavailable (copyright restrictions)
- Uniform annotations enable fair cross-lingual comparison
- mBERT pre-trained on these languages (104 total)
**Rationale for WikiANN:**
- CoNLL-2003 German unavailable (copyright restrictions)
- Uniform annotations enable fair cross-lingual comparison
- mBERT pre-trained on these languages (104 total)

## Model Architecture

**mBERT** (bert-base-multilingual-cased):
- 12 layers, 768 hidden size, 12 attention heads
- 177M parameters (all fine-tuned)
- Vocabulary: 119,547 tokens
- New classification head: 7 output classes (B-/I-PER/LOC/ORG + O)

## Training Configuration

**Hyperparameters:**
- Learning rate: 5e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW (weight_decay=0.01)

**Hardware:**
- CPU training (~4 hours)

**Evaluation:**
- Metric: Entity-level F1 (seqeval)
- Strategy: After each epoch, best 2 checkpoints retained

## Results

### Overall Performance (Combined Test Set - 15K sentences)

| Metric        | Score      |
|---------------|------------|
| **Precision** | 88.19%     |
| **Recall**    | 89.69%     |
| **F1**        | **88.93%** |

### Per-Language Performance

| Language | Precision | Recall | F1         |
|----------|-----------|--------|------------|
| German (DE) | 86.78% | 88.45% | **87.60%** |
| French (FR) | 89.01% | 90.07% | **89.53%** |
| Dutch (NL)  | 88.80% | 90.54% | **89.66%** |
| **Average** | 88.20% | 89.69% | **88.93%** |

### Comparison with English Baselines

| Dataset                  | Model     | F1 Score  | Notes                                  |
|--------------------------|-----------|-----------|----------------------------------------|
| CoNLL-2003 EN (M3)       | BiLSTM    | 84.5%     | Token-level eval, trained from scratch |
| CoNLL-2003 EN (M4)       | mBERT     | 91.46%    | Entity-level eval, pre-trained         |
| **WikiANN DE+FR+NL (M4)**| **mBERT** | **88.93%**| Entity-level eval, multilingual        |

### Training Progress

| Epoch | Train Loss | Val Loss | Precision | Recall | Val F1 |
|-------|------------|----------|-----------|---------|--------|
| 1 | 0.275 | 0.180 | 84.78% | 85.88% | 85.33% |
| 2 | 0.116 | 0.181 | 87.77% | 88.63% | 88.20% |
| 3 | 0.058 | 0.182 | 88.08% | 89.63% | **88.85%** |

**Training time**: 4 hours 12 minutes on CPU

## Key Findings

**Performance insights:**
- Strong multilingual transfer learning: 88.93% F1 across 3 languages
- Minimal validation-test gap (0.08pp) indicates excellent generalization
- Balanced performance across languages (87.60% - 89.66%), demonstrating robust cross-lingual capabilities

**Cross-module comparison:**
- **M3→M4 (English)**: +7pp improvement (BiLSTM 84.5% → mBERT 91.46%)
  - Demonstrates transformer architecture advantage and pre-training benefits
- **M4 English→Multilingual**: -2.5pp (91.46% → 88.93%)
  - Expected due to WikiANN's silver standard annotations vs CoNLL's gold standard
  - Multilingual complexity across 3 languages vs single-language optimization

**Language-specific observations:**
- French and Dutch achieve similar performance (~89.5%)
- German slightly lower (87.60%), possibly due to compound word complexity
- 2pp variance across languages indicates effective multilingual transfer

**Technical achievements:**
- Successfully fine-tuned 177M parameter model on CPU in ~4 hours
- Efficient dataset reduction strategy (20K→5K per split) maintained model quality
- mBERT's multilingual pre-training enabled strong zero-shot generalization


## Installation & Usage

**Requirements:**
```bash
pip install -r requirements.txt
```

See `requirements.txt` for package versions.

**Run notebooks:**

**Stage 1 - English Baseline:**
1. Ensure CoNLL-2003 data is in `MNER/data/raw/conll2003/`
2. Open `notebooks/M4_MNER.ipynb`
3. Run all cells (training takes ~7.5 hours on CPU)

**Stage 2 - Multilingual:**
1. Ensure WikiANN data is in `MNER/data/raw/wikiann/`
2. Open `notebooks/M4_MNER_multilingual.ipynb`
3. Run all cells (training takes ~4 hours on CPU)



## References

- **WikiANN Dataset**: [Rahimi et al. 2019](https://aclanthology.org/P19-1015/)
- **mBERT**: [Devlin et al. 2019](https://arxiv.org/abs/1810.04805)
- **seqeval**: [Entity-level evaluation for NER](https://github.com/chakki-works/seqeval)
- **Battista NER Datasets**: [GitHub Repository](https://github.com/davidsbatista/NER-datasets)

## Author

olgaanna75 - CAS NLP 2025, Module 4