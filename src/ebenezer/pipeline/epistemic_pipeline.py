import spacy
# from datasets.data_loader import load_dataset
# import numpy as np
from src.ebenezer.feature_extractors.entry import LexicalExtractor
from thinc.api import set_gpu_allocator, require_gpu
from src.ebenezer.preprocess.clause_segmentation import ClauseSegmenter
from src.ebenezer.feature_extractors.attribution_extractor import AttributionExtractor
from src.ebenezer.embeddings.span_calculation import SpanEmbeddingCalculator
from src.ebenezer.feature_assembly.assembler import FeatureAssembler


class EpistemicPipeline:
    """
    High-level pipeline to extract epistemic features from text,
    using spaCy TRF model for dependency analysis.
    """

    def __init__(self, nlp_model="en_core_web_trf"):        
        set_gpu_allocator("pytorch")
        require_gpu(0)
        
        self.clause_segmenter = ClauseSegmenter()
        self.lexical_extractor = LexicalExtractor()
        self.attribution_extractor = AttributionExtractor(
            attribution_verbs=self.lexical_extractor.attribution_verbs,
            epistemic_verbs=self.lexical_extractor.epistemic_verbs,
        )
        self.span_embedding = SpanEmbeddingCalculator()
        self.feature_assembler = FeatureAssembler()

        try: 
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Model {nlp_model} not found. Downloading...")
            from spacy.cli import download
            download(nlp_model)
            self.nlp = spacy.load(nlp_model)
                    
    def process_text(self, texts):
        """
        Process a list of texts and return spaCy Doc objects. 
        First normalsies inputs by collapsing whitespaces and newlines.
        """
        texts = [" ".join(text.split()) for text in texts]


        all_docs = []
        
        for doc in self.nlp.pipe(texts, batch_size=8):
            print(type(doc._.trf_data))
            self.lexical_extractor.annotate_trf_embeddings(doc)
            all_docs.append(doc)
            
        return all_docs
            
sample_texts = [
    '''Senior economists have reportedly warned that the proposed tax reforms could potentially trigger widespread financial instability, 
    though the Treasury Department adamantly insists there is absolutely no credible evidence supporting these alarming predictions.
    At last we don't know when the economy will recover, perhaps it will take a significant time before then.''',
]            
            
if __name__ == "__main__":
    pipeline = EpistemicPipeline()
    docs = pipeline.process_text(sample_texts)
    docs = [pipeline.lexical_extractor.annotate_doc(doc) for doc in docs]
    
    for doc in docs:
        clauses = pipeline.clause_segmenter.segment(doc)
        clauses = pipeline.attribution_extractor.extract_all(clauses, doc)

        print(f"\n{'#':<4}{'Root':<12}{'Source':<30}{'Type':<16}{'Attr':<6}{'Epc':<6}{'Dep':<10}{'Head':<14}{'Depth':<7}{'Marker':<8}  Text")
        print("-" * 156)
        for i, c in enumerate(clauses):
            print(f"{i:<4}{c['root']:<12}{str(c['source_text']):<30}{str(c['source_type']):<16}"
                f"{str(c['is_attribution_verb']):<6}{str(c['is_epistemic_verb']):<6}"
                f"{c['root_dep']:<10}{str(c['head']):<14}{c['depth']:<7}{str(c['marker']):<8}  {c['text'][:37] + '...' if len(c['text']) > 40 else c['text']}"
                )
        
        print("-" * 156)
        print("")
        print("")
        print("")
        print("-" * 156)
        
        feature_vecs = pipeline.feature_assembler.assemble_all(clauses, doc)
        for i, (c, fv) in enumerate(zip(clauses, feature_vecs)):
            print(f"{i} {c['root']:<12} feature_vec shape: {fv.shape}  "
                f"span_emb[:3]: {fv[:3].round(3)}")
            
        print("-"*156)

        print(f"\nDocument:\n{doc.text}")
        print("")
        print(f"{'Token':<12}{'Dep':<12}{'Head':<12}{'POS':<8}"
              f"{'HDG':<6}{'CRTH':<10}{'CRTL':<9}"
              f"{'EPC':<10}{'MDL':<6}{'SUBJ':<10}{'Embedding':<10}")
        print("-"*156)    
        
        for token in doc:
            emb = token._.trf_embedding
            print(
                f"{token.text:<12}{token.dep_:<12}{token.head.text:<12}{token.pos_:<8}"
                f"{str(token._.is_hedge):<6}"
                f"{str(token._.is_certainty_high):<10}"
                f"{str(token._.is_certainty_low):<9}"
                f"{str(token._.is_epistemic_verb):<10}"
                f"{str(token._.is_modal_verb):<6}"
                f"{str(token._.is_subjective_verb):<10}"
                f"{emb[:5]}"
            )                  
