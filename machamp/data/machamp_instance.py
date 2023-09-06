from typing import List, Dict, Any

import torch


class MachampInstance:
    def __init__(self,
                 full_data: List[str],
                 token_ids: torch.tensor,
                 seg_ids: torch.tensor,
                 golds: Dict[Any, torch.tensor],
                 dataset: str,
                 offsets: torch.tensor = None,
                 no_unk_subwords: List[str] = None, 
                 dataset_ids: List[int] = None):
        """

        Parameters
        ----------
        full_data: List[str]
            The (gold annotated) data read from the input file.
        token_ids: torch.tensor
            A list of token_ids, these do not include the special start/end tokens.
        seg_ids: torch.tensor
            Segment id's, also called token_type_ids in the transformers 
            library. Should have the same length as the token_ids.
        golds: Dict[str, List]
            A dictionary with annotation for each task (the name of the task is the 
            key). The Lists can have different formats, depending on the type of task.
        dataset: str
            The name of the dataset from which this instance is read.
        offsets: torch.tensor
            The offsets of the words in the wordpiece list. These can be used to align
            the wordpieces to the words and vice-versa.
        no_unk_subwords: List[str]
            The string representation of the subwords. If a subword == unk, this actually
            kept the original string, so it is not always correctly obtainable from the
            token_ids, hence we save it separately.
        """
        self.full_data = full_data
        self.token_ids = token_ids
        self.seg_ids = seg_ids
        self.golds = golds
        self.dataset = dataset
        self.offsets = offsets
        self.no_unk_subwords = no_unk_subwords
        self.dataset_ids = dataset_ids

    def __len__(self) -> int:
        """
        Defines the length of an instance as the amount of subwords

        Returns
        -------
        length
            The amount of subwords in this instance
        """
        return len(self.token_ids)

    def __str__(self) -> str:
        """
        
        """
        fullstr = 'full_data ' + str(self.full_data) + '\n'
        fullstr += "token_ids " + str(self.token_ids) + '\n'
        fullstr += 'offsets ' + str(self.offsets) + '\n'
        for task in self.golds:
            fullstr += task + ' ' + str(self.golds[task]) + '\n'
        return fullstr
