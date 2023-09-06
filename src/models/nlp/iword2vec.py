# type: ignore

from gensim.models import Word2Vec 
from multiprocessing import cpu_count
import pandas as pd
import numpy as np

class iWord2Vec():
    def __init__(self, c=5, e=64, epochs=1, source=None, destination=None, 
                                                                      seed=15):
        """ Initialize an instance of iWord2Vec.

        Parameters:
        -----------
        c : int, optional (default=5)
            The size of the context window.

        e : int, optional (default=64)
            The size of the word embeddings.

        epochs : int, optional (default=1)
            The number of training epochs.

        source : str or None, optional (default=None)
            The source file to load a pre-trained model from.

        destination : str or None, optional (default=None)
            The destination file to save the trained model.

        seed : int, optional (default=15)
            The random seed for reproducibility.

        """
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
        """
        Train the iWord2Vec model on the given corpus.

        Parameters:
        -----------
        corpus : list of list of str
            A list of sentences where each sentence is a list of words.

        save : bool, optional (default=False)
            Whether to save the trained model.

        """
        self.model = Word2Vec(sentences=corpus, vector_size=self.embedding_size, 
                              window=self.context_window, epochs=self.epochs, 
                              workers=cpu_count(), min_count=0, sg=1, 
                              negative=5, sample=0, seed=self.seed)
        if save:
            self.model.save(f'{self.destination}.model')

    def load_model(self):
        """ Load a pre-trained iWord2Vec model from a file.

        """
        self.model = Word2Vec.load(f'{self.source}.model')


    def get_embeddings(self, ips=None, emb_path=None):
        """ Get word embeddings for specific words or all words.

        Parameters:
        -----------
        ips : list of str or None, optional (default=None)
            A list of words to retrieve embeddings for. If None, retrieves embeddings for all words.

        emb_path : str or None, optional (default=None)
            The file path to save the embeddings as a CSV file.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing word embeddings.

        """
        if type(ips)==type(None):
            ips = [x for x in self.model.wv.index_to_key]
        embeddings = self.model.wv.vectors    
        embeddings = pd.DataFrame(embeddings, index=ips)

        if type(emb_path)!=type(None):
            embeddings.to_csv(emb_path)
                    
        return embeddings
    

    def update(self, corpus, save=False):
        """ Update the iWord2Vec model with additional training on a new corpus.

        Parameters:
        -----------
        corpus : list of list of str
            A list of sentences where each sentence is a list of words.

        save : bool, optional (default=False)
            Whether to save the updated model.

        """
        self.model.build_vocab(corpus, update=True, trim_rule=None)
        self.model.train(corpus, total_examples=self.model.corpus_count, 
                         epochs=self.epochs)
        if save:
            self.model.save(f'{self.destination}.model')

    def del_embeddings(self, to_drop, mname=None):
        """ Delete word embeddings for specific words.

        Parameters:
        -----------
        to_drop : list of str
            A list of words to delete from the embeddings.

        mname : str or None, optional (default=None)
            The destination file to save the model after removing embeddings.

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