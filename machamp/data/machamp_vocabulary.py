# based on /home/rob/.local/lib/python3.8/site-packages/transformers/models/bert/tokenization_bert.py

# It might be neater to put this in 2 classes?, one for the collection of vocabs, and one for a single vocab

# Some things it doesnt check for: Does the namespace already exist? Doest the token contain a newline?,
# then the file on the disk will be read wrongly (note that MaChAmp doesnt rely on this though)

import os

class MachampVocabulary:
    def __init__(self):
        """
        A class that can represent multiple vocabularies. They are kep apart by 
        a unique key (in a namespace). self.namespaces consists of a dictionary 
        with the unique keys, and dictionaries as values. These dictionary keep
        the actual labels/tokens. In self.inverse_namespace we have the same 
        structure, but use lists instead of dictionaries, so that we can also quickly
        look up words by their indices.
        """
        self.namespaces = {}
        self.inverse_namespaces = {}
        self.hasUnk = {}
        self.UNK_ID = 0
        self.UNK = '@@unkORpad@@'
        # This is perhaps not the neatest location, but it is put here for convenience, 
        # as it is used in many places, and the vocabulary is availabl in all of them
        self.pre_splits = {}

    def load_vocab(self, vocab_path: str, name: str):
        """
        Loads the vocabulary of a single namespace (i.e. text file).
        
        Parameters
        ----------
        vocab_path: str
            The path to the text file to read.
        name: str
            The unique namespace name.
        """
        vocab = {}
        inverse_vocab = []
        with open(vocab_path, "r", encoding="utf-8", errors='ignore') as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
            inverse_vocab.append(token)
        self.namespaces[name] = vocab
        self.inverse_namespaces[name] = inverse_vocab

    def load_vocabs(self, vocab_dir: str):
        """
        Load a vocabulary from a folder, expect txt files with the names of the
        namespaces in this folder, with a label/token on each line.

        Parameters
        ----------
        vocab_dir: str
            the path to the saved vocabulary.
        """
        for namespace in os.listdir(vocab_dir):
            self.load_vocab(os.path.join(vocab_dir, namespace), namespace)

    def get_unk(self, name: str):
        """
        Gets the unknown string if it exists in the specified namespace.
        
        Parameters
        ----------
        name: str
            name in the namespace.
        """
        if self.hasUnk[name]:
            return self.UNK

    def get_unk_id(self, name: str):
        """
        Gets the unknown id if it exists in the specified namespace.
        
        Parameters
        ----------
        name: str
            name in the namespace.
        """
        if self.hasUnk[name]:
            return self.UNK_ID

    def get_vocab(self, name: str):
        """
        Return the actual dictionary for a certain namespace.
        
        Parameters
        ----------
        name: str
            name in the namespace.
        
        Returns
        -------
        vocabulary: Dict[str, int]
            The dictionary containing the vocab (words, ids)
        """
        return dict(self.namespaces[name])

    def token2id(self, token: str, namespace: str, add_if_not_present: bool):
        """
        Look up a token, and return its ID.

        Parameters
        ----------
        token: str
            The token to look up.
        namespace: str
            The namespace to use.
        add_if_not_present: bool
            During the first reading of the training, we usually want to add 
            unknown labels, during prediction this is usually not the case, and
            the vocabulary should be fixed.
        
        Returns
        -------
        token_id: int
            The id of the token.
        """
        if token not in self.namespaces[namespace]:
            if add_if_not_present:
                self.namespaces[namespace][token] = len(self.inverse_namespaces[namespace])
                self.inverse_namespaces[namespace].append(token)
                return len(self.inverse_namespaces[namespace]) - 1
            else:
                return self.UNK_ID if self.hasUnk[namespace] else None
        if self.hasUnk[namespace]:
            return self.namespaces[namespace].get(token, 0)
        else:
            return self.namespaces[namespace].get(token, None)

    def id2token(self, token_id: int, namespace: str):
        """
        Look up an id, and return the corresponding token.

        Parameters
        ----------
        token_id: int
            The id of the token.
        namespace: str
            The namespace to use.
        
        Returns
        -------
        token: str
            The token to look up.
        """
        return self.inverse_namespaces[namespace][token_id]

    def create_vocab(self, name: str, has_unk: bool):
        """
        Create a new vocabulary with a unique name in the namespace.

        Parameters
        ----------
        name: str
            The name in the namespace.
        has_unk: bool
            Whether this vocabulary should have an unknown/padding token.
        """
        if name not in self.namespaces:
            self.namespaces[name] = {self.UNK: self.UNK_ID} if has_unk else {}
            self.inverse_namespaces[name] = [self.UNK] if has_unk else []
            self.hasUnk[name] = has_unk

    def save_vocabs(self, out_dir: str):
        """
        Save all the vocabs in self.namespaces, in the outDir each
        name will get its own text file.

        Parameters
        ----------
        out_dir: str
            The directory in which to write the textfiles.
        """
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        for namespace in self.namespaces:
            self.save_vocab(namespace, os.path.join(out_dir, namespace))
        open(os.path.join(out_dir, 'pre_splits_vocab'), 'w').write(str(self.pre_splits))

    def save_vocab(self, name: str, vocab_path: str):
        """
        Writes the contents of a certain namespace, as one token per line.
        
        Parameters
        ----------
        name: str
            The name in the namespace.
        vocab_path: str
            The path to write the contents to.
        """
        out_file = open(vocab_path, 'w')
        for token in self.inverse_namespaces[name]:
            out_file.write(token + '\n')
        out_file.close()
