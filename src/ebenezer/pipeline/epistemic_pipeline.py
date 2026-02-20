import spacy
# from feature_assembly.assembler import FeatureAssembler
# from datasets.data_loader import load_dataset
# import numpy as np
from src.ebenezer.feature_extractors.entry import LexicalExtractor
from thinc.api import set_gpu_allocator, require_gpu

class EpistemicPipeline:
    """
    High-level pipeline to extract epistemic features from text,
    using spaCy TRF model for dependency analysis.
    """

    def __init__(self, nlp_model="en_core_web_trf"):        
        set_gpu_allocator("pytorch")
        require_gpu(0)

        try: 
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Model {nlp_model} not found. Downloading...")
            from spacy.cli import download
            download(nlp_model)
            self.nlp = spacy.load(nlp_model)

            
            
        self.lexical_extractor = LexicalExtractor()
        
    def process_text(self, texts):
        """
        Process a list of texts and return spaCy Doc objects.
        """
        all_docs = []
        
        for doc in self.nlp.pipe(texts, batch_size=8, disable=["ner"]):
            print(type(doc._.trf_data))
            self.lexical_extractor.annotate_trf_embeddings(doc)
            all_docs.append(doc)
            
        return all_docs
            
if __name__ == "__main__":
    pipeline = EpistemicPipeline()
    sample_texts = [
        "Senior economists have reportedly warned that the proposed tax reforms could potentially trigger widespread financial instability, though the Treasury Department adamantly insists there is absolutely no credible evidence supporting these alarming predictions.",
    ]
    
    docs = pipeline.process_text(sample_texts)

    docs = [pipeline.lexical_extractor.annotate_doc(doc) for doc in docs]
    
    for doc in docs:
        print(f"\nDocument: {doc.text}")
        print(f"{'Token':<12}{'Dep':<12}{'Head':<12}{'POS':<8}"
              f"{'HDG':<6}{'CRTH':<10}{'CRTL':<9}"
              f"{'EPC':<10}{'MDL':<6}{'SUBJ':<10}")
        print("-"*90)    
        
        for token in doc:
            emb = token._.trf_embedding
            print(f"{token.text:<15} embedding[:5]: {emb[:5]}")
        
            print(f"{token.text:<12}{token.dep_:<12}{token.head.text:<12}{token.pos_:<8}"
                  f"{str(token._.is_hedge):<6}"
                  f"{str(token._.is_certainty_high):<10}"
                  f"{str(token._.is_certainty_low):<9}"
                  f"{str(token._.is_epistemic_verb):<10}"
                  f"{str(token._.is_modal_verb):<6}"
                  f"{str(token._.is_subjective_verb):<10}"
            )                  
