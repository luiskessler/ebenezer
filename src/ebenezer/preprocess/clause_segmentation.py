from spacy.tokens import Doc, Span, Token
from typing import List, Dict

class ClauseSegmenter:
    """
    Dependency-based clause segmenter.
    Identifies clause boundaries by finding verbal roots and their subtrees,
    splitting on ccomp, xcomp, advcl, relcl, parataxis.
    """

    # These dep labels anchor a new clause
    CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "parataxis"}

    # These introduce a clause but aren't the anchor themselves
    CLAUSE_MARKERS = {"mark", "cc"}

    # These deps stay attached to their head clause - never split on these
    NO_SPLIT_DEPS = {
        "amod", "advmod", "det", "prep", "pobj", "compound",
        "punct", "aux", "auxpass", "neg", "poss", "quantmod",
        "nummod", "appos", "attr", "expl", "nsubj", "dobj",
        "nsubjpass", "agent", "acomp"
    }

    def __init__(self):
        Span.set_extension("clause_root", default=None, force=True)
        Span.set_extension("clause_dep", default=None, force=True)
        Span.set_extension("clause_head", default=None, force=True)
        Span.set_extension("clause_depth", default=0, force=True)

    def _get_clause_tokens(self, verb_token: Token, claimed: set) -> List[Token]:
        tokens = []

        def walk(token):
            if token.i in claimed:
                return
            if token != verb_token and token.dep_ in self.CLAUSE_DEPS and token.pos_ in {"VERB", "AUX"}:
                return
            claimed.add(token.i)
            tokens.append(token)
            for child in token.children:
                walk(child)

        walk(verb_token)
        return sorted(tokens, key=lambda t: t.i)

    def _find_clause_roots(self, doc: Doc) -> List[Token]:
        """
        Find all tokens that anchor a clause, ordered by position.
        """
        roots = []
        for token in doc:
            if token.dep_ in self.CLAUSE_DEPS and token.pos_ in {"VERB", "AUX"}:
                roots.append(token)
        return roots

    def _compute_depth(self, token: Token) -> int:
        """
        Count how many clause boundaries exist between this token and ROOT.
        """
        depth = 0
        current = token
        while current.head != current:
            if current.dep_ in self.CLAUSE_DEPS:
                depth += 1
            current = current.head
        return depth
    
    def segment(self, doc: Doc) -> List[Dict]:
        clause_roots = self._find_clause_roots(doc)
        claimed = set()
        clauses = []

        clause_roots_sorted = sorted(
            clause_roots,
            key=lambda t: self._compute_depth(t),
            reverse=True
        )

        for verb in clause_roots_sorted:
            tokens = self._get_clause_tokens(verb, claimed)

            if not tokens:
                continue

            start = tokens[0].i
            end = tokens[-1].i + 1
            span = doc[start:end]

            span._.clause_root = verb
            span._.clause_dep = verb.dep_
            span._.clause_head = verb.head if verb.dep_ != "ROOT" else None
            span._.clause_depth = self._compute_depth(verb)

            clauses.append({
                "span": span,
                "text": " ".join(t.text for t in tokens if t.pos_ != "PUNCT"),
                "root": verb.text,
                "root_token": verb,
                "root_dep": verb.dep_,
                "head": verb.head.text if verb.dep_ != "ROOT" else None,
                "depth": span._.clause_depth,
                "marker": self._get_marker(verb),
            })

        clauses.sort(key=lambda c: c["span"].start)
        return clauses

    def _get_marker(self, verb: Token) -> str | None:
        """
        Get the subordinating conjunction or marker introducing this clause.
        e.g. 'that', 'though', 'which'
        """
        for child in verb.children:
            if child.dep_ in self.CLAUSE_MARKERS:
                return child.text
        return None