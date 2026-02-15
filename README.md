# WORK IN PROGRESS!!

# Ebenezer: Neural Epistemic Stance Classification

Research implementation of the epistemic stance classifier described in 
"Ebenezer: Neural Epistemic Stance Classification for News Analysis" ([arXiv link]).

Ebenezer classifies sentences in news articles into four epistemic categories:
- **Claim**: Factual assertions ("The GDP grew by 3%")
- **Opinion**: Subjective evaluations ("This policy is misguided")
- **Speculative**: Hedged or uncertain statements ("Experts believe this may lead to...")
- **Neutral**: Contextual or background information ("The meeting took place on Tuesday")

## Production Use

Ebenezer was developed for [Knovolo](https://www.knovolo.com), a news analysis platform. Knovolo's production implementation is proprietary and includes:
- Optimized inference in Rust
- Proprietary training data (2,000+ annotated news sentences)
- Integration with Knovolo's full news analysis pipeline (geoparsing, entity extraction, narrative sentiment, ideological vector analysis)

## Installation
```bash
pip install ebenezer
# or
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- transformers
- spacy >= 3.4

## Quick Start
```python
from ebenezer import EbenezerClassifier

model = EbenezerClassifier.load("ebenezer-base")
text = "Officials believe the casualties may exceed 100."

result = model.predict(text)
print(result)
# {'label': 'speculative', 'confidence': 0.89}
```

## Model Details

See the paper for full details: [arXiv link] or [knovolo.com/publications/ebenezer](https://www.knovolo.com/publications/ebenezer)

**Architecture:** Hybrid neural model combining:
- Contextualized embeddings (spaCy RoBERTa)
- Linguistic features (modal verbs, hedge words, attribution patterns)
- Dense classification layers

**Training data:** Unified dataset from factuality, subjectivity, and hedge detection corpora (FactBank, MPQA, BioScope, ClaimBuster, TR News, GeoWebNews)

**Performance:**
- Overall accuracy: [To be tested]
- Cross-domain evaluation: See paper Table 1

## Citation

If you use Ebenezer in academic work, please cite:
```bibtex
@article{kessler2026ebenezer,
      title={Ebenezer: Neural Epistemic Stance Classification for News Analysis}, 
      author={Luis Kessler},
      year={2026},
      journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

Commercial users should include attribution in documentation or UI.

## License

MIT License - Copyright (c) 2026 Luis Kessler

See [LICENSE](LICENSE) for full terms.

## Contributing

Contributions welcome! Please open an issue before submitting PRs.

Areas for improvement:
- Additional language support
- More training data
- Domain adaptation for scientific/legal text

## Acknowledgments

This research was conducted as part of the Knovolo project. Special thanks to [any collaborators, annotators, funding sources].

---

**Related Projects:**
- [Knovolo](https://www.knovolo.com) - Geospatial intelligence infrastructure for event analysis
