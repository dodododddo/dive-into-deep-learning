'''实现一个BPE/Wordpiece分词器'''

import re
import toolz
from collections import Counter
import torch

def wordpunct_tokenize(text):
    pattern = r"\w+|[^\w\s]+"
    regexp = re.compile(pattern, flags=re.UNICODE | re.MULTILINE | re.DOTALL)
    return regexp.findall(text)

class BPETokenizer:
    special = ["<unk>", "<pad>", "<end>", "<mask>"]
    
    def __init__(self, vocab_size = 1000, lowercase = True, basic_tokenizer = wordpunct_tokenize):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.basic_tokenizer = basic_tokenizer
        
    def fit(self, corpus, max_steps, out_path = 'vocab.txt'):
        if self.lowercase:
            corpus = [c.lower() for c in corpus]
        word_corpus = Counter([tuple(data) + ("</w>",) for data in toolz.concat(map(self.basic_tokenizer, corpus))])
        vocab = self._count_vocab(word_corpus)
        
        for _ in range(max_steps):
            word_corpus, bi_cnt = self._fit_step(word_corpus)
            vocab = self._count_vocab(word_corpus)
            if len(vocab) > self.vocab_size or bi_cnt < 0:
                break
            
        for s in self.special:
            if s not in vocab:
                vocab.insert(0, (s, 99999))
            with open(out_path, 'w') as f:
                f.write('\n'.join([w for w, _ in vocab]))
        self.vocab = [token for token, _ in vocab]
        return vocab
        
    
    def _count_vocab(self, word_corpus):
        r = Counter([data for data in toolz.concat([word * cnt for word, cnt in word_corpus.items()])])
        r = sorted(r.items(), key=lambda x: -x[1])
        return r
    
    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()

        ############### 以步长1，窗口尺寸2，在每个单词上滚动，统计二元组频次 ###############
        for tokens, count in word_corpus.items():
            if len(tokens) < 2: 
                continue
            for bigram in toolz.sliding_window(ngram, tokens):
                bigram_counter[bigram] += count

        ############### 选出频次最大的二元组 ###############
        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=bigram_counter.get)
        else:
            return word_corpus, -1
        bi_cnt = bigram_counter.get(max_bigram)

        ############### 从corpus中将最大二元组出现的地方替换成一个token ###############
        words_tokens = list(word_corpus.keys())
        for tokens in words_tokens:
            new_tokens = tuple(' '.join(tokens).replace(' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if new_tokens != tokens:
                word_corpus[new_tokens] = word_corpus[tokens]
                word_corpus.pop(tokens)
        return word_corpus, bi_cnt
    
    def tokenize(self, text):
        tokens = []
        
        # 初步分词
        if self.lowercase:
            text = text.lower()
        for token in self.basic_tokenizer(text):
            new_tokens = []
            token = list(token) + ['</w>']
            # 寻找最长合法子词
            start, end = 0, len(token)
            while start < end:
                # 左闭右开，恰好不会考虑到'</w>'
                sub_token = ''.join(token[start : end])
                if sub_token in self.vocab:
                    new_tokens.append(sub_token)
                    start = end
                    end = len(token)
                elif end - start == 1:
                    new_tokens.append('<unk>')
                    start = end
                    end = len(token)
                else:
                    end -= 1
            tokens.extend(new_tokens)
        
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self._token2id(token) for token in tokens]
        return ids
    
    def decode(self, ids):
        tokens = [self._id2token(id) for id in ids]
        sentence = ''.join(tokens).replace('</w>', ' ')
        return sentence
    
    def encode2tensor(self, text):
        ids = self.encode(text)
        result = torch.zeros(len(ids), len(self.vocab))
        index = torch.LongTensor(ids).unsqueeze(1)
        # 待改进，暂时没找到合适的赋值方法
        result = result.scatter(1, index, 1)
        return result
        
            
    def _token2id(self, token):
        if token in self.vocab:
            return self.vocab.index(token)
        else:
            return self.vocab.index('<unk>')
        
    def _id2token(self, id):
        return self.vocab[id]


class WordPieceTokenizer(BPETokenizer):
    def _fit_step(self, word_corpus):
        ngram = 2
        bigram_counter = Counter()
        unigram_counter = Counter()
        
        ############### 以步长1，窗口尺寸2，在每个单词上滚动，统计二元组频次 ###############
        for token, count in word_corpus.items():
            for c in token:
                unigram_counter[c] += count
            if len(token) < 2: continue
            for bigram in toolz.sliding_window(ngram, token):
                bigram_counter[bigram] += count
        
        ############### 选出频次最大的二元组 ###############
        if len(bigram_counter) > 0:
            max_bigram = max(bigram_counter, key=lambda x: bigram_counter.get(x) / (unigram_counter.get(x[0]) * unigram_counter.get(x[1])))
        else:
            return word_corpus, -1
        bi_cnt = max(bigram_counter.values())
        
        words_tokens = list(word_corpus.keys())
        ############### 从corpus中将最大二元组出现的地方替换成一个token ###############
        for token in words_tokens:
            _new_token = tuple(' '.join(token).replace(' '.join(max_bigram), ''.join(max_bigram)).split(' '))
            if _new_token != token:
                word_corpus[_new_token] = word_corpus[token]
                word_corpus.pop(token)
        return word_corpus, bi_cnt


def load_corpus(path: str):
    with open(path, 'r') as f:
        return f.read().split('\n')
    
    
    
if __name__ == '__main__':
    corpus = load_corpus('/home/jr/dl/dl_ch10/corpus.txt')
    tokenizer = WordPieceTokenizer()
    tokenizer.fit(corpus, 2000)

    text = '''
           The reasons for this trend may involve the recognition 
           that a young adult who passes directly from school 
           to university is rather restricted in terms of general 
           knowledge and experience of the world. By contrast, 
           those who have spent some time earning a living or 
           travelling to other places, have a broader view of life and better personal resources to draw on. 
           They tend to be more independent, which is a very important factor in academic study and research, 
           as well as giving them an advantage in terms of coping with the challenges of student life.
           '''
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    input = tokenizer.encode2tensor(text)
    regen_text = tokenizer.decode(ids)
    print(tokens)
    print(ids)
    print(input[0][10] == 1)
    print(regen_text)

