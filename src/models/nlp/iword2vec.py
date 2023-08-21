# type: ignore

from gensim.models import Word2Vec 
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

class iWord2Vec():
    """_summary_

    Parameters
    ----------
    c : int, optional
        _description_, by default 5
    e : int, optional
        _description_, by default 64
    epochs : int, optional
        _description_, by default 1
    source : _type_, optional
        _description_, by default None
    destination : _type_, optional
        _description_, by default None
    seed : int, optional
        _description_, by default 15
    """
    def __init__(self, c=5, e=64, epochs=1, source=None, destination=None, 
                                                                      seed=15):
        self.context_window = c
        self.embedding_size = e
        self.epochs = epochs
        self.seed = seed
        
        self.model = None

        self.source = source
        self.destination = destination

        if type(source) != type(None):
            self.load_model()            
                
    def train(self, corpus, save=False):
        """_summary_

        Parameters
        ----------
        corpus : _type_
            _description_
        save : bool, optional
            _description_, by default False
        """
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size, 
                              window=self.context_window, epochs=self.epochs, 
                              workers=cpu_count(), min_count=0, sg=1, 
                              negative=5, sample=0, seed=self.seed)
        if save:
            self.model.save(f'{self.destination}.model')

    def load_model(self):
        """_summary_
        """
        self.model = Word2Vec.load(f'{self.source}.model')


    def get_embeddings(self, ips=None, emb_path=None):
        """_summary_

        Parameters
        ----------
        ips : _type_, optional
            _description_, by default None
        emb_path : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        if type(ips)==type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors    
        embeddings = pd.DataFrame(embeddings, index=ips)

        if type(emb_path)!=type(None):
            embeddings.to_csv(emb_path)
                    
        return embeddings
    

    def update(self, corpus, save=False):
        """_summary_

        Parameters
        ----------
        corpus : _type_
            _description_
        save : bool, optional
            _description_, by default False
        """
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count, 
                         epochs=self.epochs)
        if save:
            self.model.save(f'{self.destination}.model')

    def del_embeddings(self, to_drop, mname=None):
        """_summary_

        Parameters
        ----------
        to_drop : _type_
            _description_
        mname : _type_, optional
            _description_, by default None
        """
        idx = np.isin(self.model.wv.index_to_key, to_drop)
        idx = np.where(idx==True)[0]
        self.model.wv.index_to_key = list(np.delete(self.model.wv.index_to_key, 
                                                                  idx, axis=0))
        self.model.wv.vectors = np.delete(self.model.wv.vectors, idx, axis=0)
        self.model.syn1neg = np.delete(self.model.syn1neg, idx, axis=0)
        list(map(self.model.wv.key_to_index.__delitem__, 
                    filter(self.model.wv.key_to_index.__contains__,to_drop)))


        for i, word in enumerate(self.model.wv.index_to_key):
            self.model.wv.key_to_index[word] = i

        if type(mname)!=type(None):
            self.model.save(f'{mname}.model')