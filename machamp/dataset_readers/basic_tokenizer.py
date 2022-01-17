from typing import List, Optional

from overrides import overrides

from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer


@Tokenizer.register("bert_basic_tokenizer")
class BertBasicTokenizer(Tokenizer):
    """
    Uses the BertBasicTokenizer to split punctuation from tokens
    """
    def __init__(
        self, 
    ) -> None:
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False,strip_accents=False)

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        # TODO do not use hardcoded? this is not even correct?
        return [Token('[CLS]')] + tokens1 + (tokens2 or []) + [Token('[SEP]')]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in self.basic_tokenizer.tokenize(text)]
