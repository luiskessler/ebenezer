import numpy as np
import spacy
from spacy.tokens import Token, Doc
from pathlib import Path

LEXICON_PATH = Path(__file__).parent.parent / "config" / "lexicons"

print(LEXICON_PATH)

class LexicalExtractor:
    """
    Extracts lexical epistemic features from spaCy Doc objects using provided lexicons.
    """
    
    def __init__(self):
        self.hedges = self._load_lexicon("hedge_words.txt")
        self.attribution_phrases = self._load_lexicon("attribution_phrases.txt")
        self.attribution_verbs = self._load_lexicon("attribution_verbs.txt")
        self.certainty_high = self._load_lexicon("certainty_high.txt")
        self.certainty_low = self._load_lexicon("certainty_low.txt")
        self.epistemic_verbs = self._load_lexicon("epistemic_verbs.txt")
        self.modal_verbs = self._load_lexicon("modal_verbs.txt")
        self.subjective_verbs = self._load_lexicon("subjective_verbs.txt")
        
        # Register extensions — force=True avoids errors on re-import/reload
        Token.set_extension("is_hedge", default=False, force=True)
        Token.set_extension("is_attribution_phrase", default=False, force=True)
        Token.set_extension("is_attribution_verb", default=False, force=True)
        Token.set_extension("is_certainty_high", default=False, force=True)
        Token.set_extension("is_certainty_low", default=False, force=True)
        Token.set_extension("is_epistemic_verb", default=False, force=True)
        Token.set_extension("is_modal_verb", default=False, force=True)
        Token.set_extension("is_subjective_verb", default=False, force=True)
        Token.set_extension("trf_embedding", default=None, force=True)
        
    def _load_lexicon(self, filename):
        path = LEXICON_PATH / filename
        if path.exists():
            return set(line.strip().lower() for line in path.read_text().splitlines() if line.strip())
        return set()
    
    def annotate_doc(self, doc: Doc):
        """
        Annotate tokens in a spaCy Doc with lexical features.
        """
        for token in doc:
            token_lower = token.text.lower()
            token._.is_hedge = token_lower in self.hedges
            token._.is_attribution_phrase = token_lower in self.attribution_phrases
            token._.is_attribution_verb = token_lower in self.attribution_verbs
            token._.is_certainty_high = token_lower in self.certainty_high
            token._.is_certainty_low = token_lower in self.certainty_low
            token._.is_epistemic_verb = token_lower in self.epistemic_verbs
            token._.is_modal_verb = token_lower in self.modal_verbs
            token._.is_subjective_verb = token_lower in self.subjective_verbs
            
        return doc

    def annotate_trf_embeddings(self, doc: Doc):
        """Attach transformer embeddings to each token (spacy-curated-transformers v3.7+ API)."""
        trf_data = doc._.trf_data

        if trf_data is None:
            return doc

        last_hidden = trf_data.last_hidden_layer_state

        piece_vectors = last_hidden.dataXd
        lengths = last_hidden.lengths

        hidden_dim = piece_vectors.shape[-1]
        offset = 0

        for i, token in enumerate(doc):
            n_pieces = int(lengths[i])

            if n_pieces == 0:
                token._.trf_embedding = np.zeros(hidden_dim)
            else:
                token_pieces = piece_vectors[offset : offset + n_pieces]
                token._.trf_embedding = token_pieces.mean(axis=0)

            offset += n_pieces

        return doc
    
    def extract_features(self, doc: Doc):
        """
        Return a simple dict of counts for each lexical feature in the doc.
        """
        return {
            "hedges": sum(token._.is_hedge for token in doc),
            "attribution_phrases": sum(token._.is_attribution_phrase for token in doc),
            "attribution_verbs": sum(token._.is_attribution_verb for token in doc),
            "certainty_high": sum(token._.is_certainty_high for token in doc),
            "certainty_low": sum(token._.is_certainty_low for token in doc),
            "epistemic_verbs": sum(token._.is_epistemic_verb for token in doc),
            "modal_verbs": sum(token._.is_modal_verb for token in doc),
            "subjective_verbs": sum(token._.is_subjective_verb for token in doc),
        }