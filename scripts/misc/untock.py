from allennlp.data.tokenizers import PretrainedTransformerTokenizer


data = [    0,    22,   412,   781,   189,   126,    21,  2554,  1097,   205,
            42, 16015,   282, 11496,  1832,    15,   223, 84722, 21515,  1097,
           205,    15,     1]


def unTok(tokenizer, data):
    tokenizer = PretrainedTransformerTokenizer(tokenizer)
    print(len(data))
    for wordpieceIdx in data:
        print(tokenizer.tokenizer._convert_id_to_token(wordpieceIdx), end= ' ')
    print()

#unTok('bert-base-multilingual-cased', data1)
#unTok('bert-base-multilingual-cased', data2)
unTok('xlm-mlm-tlm-xnli15-1024', data)


xlm15
{'tokens': {'tokens': tensor([[ 3, 24, 25, 26, 27, 28, 29, 10, 30, 31, 32, 33, 34, 35,  2,  4,  0,  0,
          0,  0,  0,  0],
        [ 3, 12, 13, 14, 15, 16, 17,  8,  9, 18, 10, 11, 19, 20,  2, 21, 22, 23,
          8,  9,  2,  4]], device='cuda:0')}}
mbert
{'tokens': {'tokens': tensor([[ 3, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  2,  4,  0,  0,  0,  0],
        [ 3, 10, 11, 12, 13,  8, 14, 15,  9, 16, 17,  2, 18, 19, 20,  8,  2,  4]],
       device='cuda:0')}}



