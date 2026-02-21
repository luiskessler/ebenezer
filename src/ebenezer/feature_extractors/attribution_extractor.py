from spacy.tokens import Doc, Token, Span
from typing import List, Dict, Optional

SOURCE_NAMED_ENTITY = "NAMED_ENTITY"      # NER-identified: ORG, PERSON, GPE etc.
SOURCE_NOMINAL = "NOMINAL"                 # Generic noun chunk: "senior economists"
SOURCE_PRONOMINAL = "PRONOMINAL"           # Pronoun subject: "they", "he"
SOURCE_JOURNALIST = "JOURNALIST"           # ROOT clause, no attribution verb
SOURCE_PASSIVE = "PASSIVE"                 # Passive construction, no overt subject

NER_SOURCE_TYPES = {"ORG", "PERSON", "GPE", "NORP", "FAC", "EVENT"}

JOURNALIST_PRONOUNS = {"it", "there", "we", "our"}

class AttributionExtractor:
    """
    Identifies attribution patterns by noun chunk extraction,
    lexicon-based attribution words and NER.

    For each clause root:
        -> find nsubj dependency
            -> if nsubj token is inside a doc.ent span -> NAMED_ENTITY (ORG/PERSON/GPE...)
            -> elif nsubj token is inside a noun_chunk -> NOMINAL
            -> elif nsubj is pronoun -> PRONOMINAL (needs resolution)
            -> no nsubj found -> JOURNALIST or PASSIVE
    """

    def __init__(self, attribution_verbs: set, epistemic_verbs: set):
        self.attribution_verbs = attribution_verbs
        self.epistemic_verbs = epistemic_verbs

    def _find_nsubj(self, verb: Token) -> Optional[Token]:
        """Find the nominal subject of a verb."""
        for child in verb.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return child
        return None

    def _get_entity_span(self, token: Token, doc: Doc) -> Optional[Span]:
        """Return the NER span containing this token, if any."""
        for ent in doc.ents:
            if ent.start <= token.i < ent.end and ent.label_ in NER_SOURCE_TYPES:
                return ent
        return None

    def _get_noun_chunk(self, token: Token, doc: Doc) -> Optional[Span]:
        """Return the noun chunk containing this token, if any."""
        for chunk in doc.noun_chunks:
            if chunk.start <= token.i < chunk.end:
                return chunk
        return None

    def _is_passive(self, verb: Token) -> bool:
        """Check if the verb is in a passive construction."""
        for child in verb.children:
            if child.dep_ == "auxpass":
                return True
        return False

    def _is_attribution_verb(self, verb: Token) -> bool:
        return verb.lemma_.lower() in self.attribution_verbs

    def _is_epistemic_verb(self, verb: Token) -> bool:
        return verb.lemma_.lower() in self.epistemic_verbs

    def extract(self, clause: Dict, doc: Doc) -> Dict:
        """
        Extract attribution information from a clause dict
        (as produced by ClauseSegmenter).
        
        Returns the clause dict enriched with attribution fields.
        """
        verb: Token = clause["root_token"]
        nsubj = self._find_nsubj(verb)

        source_span = None
        source_type = None
        source_text = None

        if nsubj is not None:
            if nsubj.lower_ in JOURNALIST_PRONOUNS and nsubj.dep_ in {"nsubj", "expl"} and clause["depth"] == 0:
                source_type = SOURCE_JOURNALIST
            else:
                entity = self._get_entity_span(nsubj, doc)

                if entity is not None:
                    source_span = entity
                    source_type = SOURCE_NAMED_ENTITY
                    source_text = entity.text

                else:
                    chunk = self._get_noun_chunk(nsubj, doc)
                    if chunk is not None:
                        source_span = chunk
                        source_text = chunk.text
                        if nsubj.pos_ == "PRON":
                            source_type = SOURCE_PRONOMINAL
                        else:
                            source_type = SOURCE_NOMINAL

                    elif nsubj.pos_ == "PRON":
                        source_type = SOURCE_PRONOMINAL
                        source_text = nsubj.text

        else:
            if self._is_passive(verb):
                source_type = SOURCE_PASSIVE
            else:
                source_type = SOURCE_JOURNALIST

        return {
            **clause,
            "source_text": source_text,
            "source_type": source_type,
            "source_ner_label": source_span.label_ if (source_span and hasattr(source_span, "label_")) else None,
            "is_attribution_verb": self._is_attribution_verb(verb),
            "is_epistemic_verb": self._is_epistemic_verb(verb),
        }

    def _build_parent_map(self, clauses: List[Dict]) -> Dict[str, Dict]:
        """Map each clause root text to its clause dict for parent lookup."""
        return {c["root"]: c for c in clauses}

    def _inherit_sources(self, clauses: List[Dict]) -> List[Dict]:
        """
        Post-process: clauses with no attribution/epistemic verb
        inherit source from their parent clause.
        """
        parent_map = self._build_parent_map(clauses)

        for clause in clauses:
            if clause["source_type"] == SOURCE_JOURNALIST or (
                not clause["is_attribution_verb"] and 
                not clause["is_epistemic_verb"] and
                clause["head"] is not None
            ):
                parent = parent_map.get(clause["head"])
                if parent and parent["source_text"] is not None:
                    clause["source_text"] = parent["source_text"]
                    clause["source_type"] = parent["source_type"]
                    clause["source_ner_label"] = parent["source_ner_label"]
                    clause["inherited"] = True
                    continue

            clause["inherited"] = False
        return clauses

    def extract_all(self, clauses: List[Dict], doc: Doc) -> List[Dict]:
        clauses = [self.extract(clause, doc) for clause in clauses]
        clauses = self._inherit_sources(clauses)
        return clauses