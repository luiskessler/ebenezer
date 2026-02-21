import numpy as np
from spacy.tokens import Span

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
POS_WEIGHTS = {"VERB": 2.0, "NOUN": 2.0, "PROPN": 2.0, "ADJ": 1.5, "ADV": 1.0}


class SpanEmbeddingCalculator:
    """
    Weighted mean pool over content tokens to get a span embedding.
    Verbs and nouns weighted highest as primary semantic anchors. 
    Falls back to all non-punct tokens.
    Potential improvement: IDF-weighted mean.
    """

    def get_span_embedding(self, span: Span) -> np.ndarray:
        vecs, weights = [], []
        for t in span:
            if t._.trf_embedding is not None and t.pos_ in POS_WEIGHTS:
                vecs.append(t._.trf_embedding)
                weights.append(POS_WEIGHTS[t.pos_])
        if not vecs:
            vecs = [
                t._.trf_embedding for t in span
                if t._.trf_embedding is not None 
                and t.pos_ not in {"PUNCT", "SPACE"}
            ]
            return np.mean(vecs, axis=0) if vecs else np.zeros(768)
        
        weights = np.array(weights)
        return np.average(vecs, axis=0, weights=weights)