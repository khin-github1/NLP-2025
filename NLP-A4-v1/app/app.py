from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

import pickle
import re
from random import *
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import datasets
from sklearn.metrics.pairwise import cosine_similarity
import spacy


app = Flask(__name__)
# en for English
from datasets import load_dataset
dataset = load_dataset('bookcorpus', split='train[:1%]')
dataset = dataset.select(range(100000))
sentences = dataset['text']
text = [x.lower() for x in sentences] #lower case
text = [re.sub("[.,!?\\-]", '', x) for x in text]


word_list = list(set(" ".join(text).split()))
word2id   = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word2id[w] = i + 4 #reserve the first 0-3 for CLS, PAD
    id2word    = {i:w for i, w  in enumerate(word2id)}
    vocab_size = len(word2id)
    
token_list = list()
for sentence in text:
    arr = [word2id[word] for word in sentence.split()]
    token_list.append(arr)

batch_size = 6
max_mask   = 5 #even though it does not reach 15% yet....maybe you can set this threshold
max_len    = 200 #maximum length that my transformer will accept.....all sentence will be padded

n_layers = 6    # number of Encoder of Encoder Layer
n_heads  = 8    # number of heads in Multi-Head Attention
d_model  = 768  # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        #x, seg: (bs, len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (len,) -> (bs, len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn 
    
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn       = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        
        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

# tokenize the sentence of model 1
def tokenize_sentence_model1(sentence_a, sentence_b):
    lst_input_ids_premise = []
    lst_input_ids_hypothesis = []
    lst_masked_tokens_premise = []
    lst_masked_pos_premise = []
    lst_masked_tokens_hypothesis = []
    lst_masked_pos_hypothesis = []
    lst_segment_ids = []
    lst_attention_premise=[]
    lst_attention_hypothesis=[]
    max_seq_length = 200
    seed(55) 

    tokens_premise, tokens_hypothesis            = [word2id[word] if word in word_list else len(word_list) for word in sentence_a.split()], \
                                                    [word2id[word] if word in word_list else len(word_list) for word in sentence_b.split()]
    
    input_ids_premise = [word2id['[CLS]']] + tokens_premise + [word2id['[SEP]']]
    input_ids_hypothesis = [word2id['[CLS]']] + tokens_hypothesis + [word2id['[SEP]']]
    
    #2. segment embedding 
    segment_ids = [0] * max_seq_length
     #3 masking
    n_pred_premise = min(max_mask, max(1, int(round(len(input_ids_premise) * 0.15))))

    #get all the pos excluding CLS and SEP
    candidates_masked_pos_premise = [i for i, token in enumerate(input_ids_premise) if token != word2id['[CLS]'] 
                                 and token != word2id['[SEP]']]
    shuffle(candidates_masked_pos_premise)
    masked_tokens_premise, masked_pos_premise = [], [] #compare the output with masked_tokens
    #simply loop and mask accordingly
    for pos in candidates_masked_pos_premise[:n_pred_premise]:
        masked_pos_premise.append(pos)
        masked_tokens_premise.append(input_ids_premise[pos])
           
        if random() < 0.1:  #10% replace with random token
            index = randint(0, vocab_size - 1)
            input_ids_premise[pos] = word2id[id2word[index]]
        elif random() < 0.8:  #80 replace with [MASK]
            input_ids_premise[pos] = word2id['[MASK]']
        else: 
            pass

    n_pred_hypothesis = min(max_mask, max(1, int(round(len(input_ids_hypothesis) * 0.15))))
    #get all the pos excluding CLS and SEP
    candidates_masked_pos_hypothesis = [i for i, token in enumerate(input_ids_hypothesis) if token != word2id['[CLS]'] 
                                 and token != word2id['[SEP]']]
    shuffle(candidates_masked_pos_hypothesis)
    masked_tokens_hypothesis, masked_pos_hypothesis = [], [] #compare the output with masked_tokens
    #simply loop and mask accordingly
    for pos in candidates_masked_pos_hypothesis[:n_pred_hypothesis]:
        masked_pos_hypothesis.append(pos)
        masked_tokens_hypothesis.append(input_ids_hypothesis[pos])
        if random() < 0.1:  #10% replace with random token
            index = randint(0, vocab_size - 1)
            input_ids_hypothesis[pos] = word2id[id2word[index]]
        elif random() < 0.8:  #80 replace with [MASK]
            input_ids_hypothesis[pos] = word2id['[MASK]']
        else: 
            pass
        
        

    #4. pad the sentence to the max length
    n_pad_premise = max_seq_length - len(input_ids_premise)
    input_ids_premise.extend([0] * n_pad_premise)
        
    #5. pad the mask tokens to the max length
    if max_mask > n_pred_premise:
        n_pad_premise = max_mask - n_pred_premise
        masked_tokens_premise.extend([0] * n_pad_premise)
        masked_pos_premise.extend([0] * n_pad_premise)
        attention_premise = [1]*n_pred_premise+[0]*(n_pad_premise)
            

    #4. pad the sentence to the max length
    n_pad_hypothesis = max_seq_length - len(input_ids_hypothesis)
    input_ids_hypothesis.extend([0] * n_pad_hypothesis)
        
    #5. pad the mask tokens to the max length
    if max_mask > n_pred_hypothesis:
        n_pad_hypothesis = max_mask - n_pred_hypothesis
        masked_tokens_hypothesis.extend([0] * n_pad_hypothesis)
        masked_pos_hypothesis.extend([0] * n_pad_hypothesis)
        attention_hypothesis = [1]*n_pred_hypothesis+[0]*(n_pad_hypothesis)
        
        

    lst_input_ids_premise.append(input_ids_premise)
    lst_input_ids_hypothesis.append(input_ids_hypothesis)
    lst_segment_ids.append(segment_ids)
    lst_masked_tokens_premise.append(masked_tokens_premise)
    lst_masked_pos_premise.append(masked_pos_premise)
    lst_masked_tokens_hypothesis.append(masked_tokens_hypothesis)
    lst_masked_pos_hypothesis.append(masked_pos_hypothesis)
    lst_attention_premise.append(attention_premise)
    lst_attention_hypothesis.append(attention_hypothesis)


    return {
        "premise_input_ids": lst_input_ids_premise,
        "premise_pos_mask":lst_masked_pos_premise,
        "hypothesis_input_ids": lst_input_ids_hypothesis,
        "hypothesis_pos_mask": lst_masked_pos_hypothesis,
        "segment_ids": lst_segment_ids,
        "attention_premise": lst_attention_premise,
        "attention_hypothesis": lst_attention_hypothesis,
        
    }

def calculate_similarity_model1(model, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs = tokenize_sentence_model1(sentence_a, sentence_b)
    
    # Move input IDs and attention masks to the active device
    inputs_ids_a = torch.tensor(inputs['premise_input_ids'])
    pos_mask_a = torch.tensor(inputs['premise_pos_mask'])
    attention_a = torch.tensor(inputs['attention_premise'])
    inputs_ids_b = torch.tensor(inputs['hypothesis_input_ids'])
    pos_mask_b = torch.tensor(inputs['hypothesis_pos_mask'])
    attention_b = torch.tensor(inputs['attention_hypothesis'])
    segment = torch.tensor(inputs['segment_ids'])

    # Extract token embeddings from BERT
    u,_ = model(inputs_ids_a, segment, pos_mask_a)  # all token embeddings A = batch_size, seq_len, hidden_dim
    v,_ = model(inputs_ids_b, segment, pos_mask_b) # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score
model1 = BERT()
model1.load_state_dict(torch.load('models/best-bert-model.pt'))
# classifier_head has shape (vocab_size*3,3)
classifier_head = torch.nn.Linear(23068*3, 3)

optimizer = torch.optim.Adam(model1.parameters(), lr=2e-5)
optimizer_classifier = torch.optim.Adam(classifier_head.parameters(), lr=2e-5)

criterion = nn.CrossEntropyLoss()
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def predict_nli_and_similarity(model, sentence_a, sentence_b, device):
    # Tokenization and embedding extraction
    inputs = tokenize_sentence_model1(sentence_a, sentence_b)
    
    # Move input IDs and attention masks to the device
    inputs_ids_a = torch.tensor(inputs['premise_input_ids']).to(device)
    pos_mask_a = torch.tensor(inputs['premise_pos_mask']).to(device)
    attention_a = torch.tensor(inputs['attention_premise']).to(device)
    inputs_ids_b = torch.tensor(inputs['hypothesis_input_ids']).to(device)
    pos_mask_b = torch.tensor(inputs['hypothesis_pos_mask']).to(device)
    attention_b = torch.tensor(inputs['attention_hypothesis']).to(device)
    segment = torch.tensor(inputs['segment_ids']).to(device)

    # Extract token embeddings
    with torch.no_grad():
        u, _ = model(inputs_ids_a, segment, pos_mask_a)
        v, _ = model(inputs_ids_b, segment, pos_mask_b)

    # Mean pooling
    u = mean_pool(u, attention_a)
    v = mean_pool(v, attention_b)

    # Convert to numpy for cosine similarity
    u_np = u.cpu().numpy().reshape(-1)
    v_np = v.cpu().numpy().reshape(-1)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u_np.reshape(1, -1), v_np.reshape(1, -1))[0, 0]

    # Compute NLI classification
    uv_abs = torch.abs(u - v)
    x = torch.cat([u, v, uv_abs], dim=-1)

    with torch.no_grad():
        logits = classifier_head(x)
        probabilities = F.softmax(logits, dim=-1)

    # NLI labels: Entailment (0), Neutral (1), Contradiction (2)
    labels = ["Entailment", "Neutral", "Contradiction"]
    nli_result = labels[torch.argmax(probabilities).item()]

    return {"similarity_score": similarity_score, "nli_label": nli_result} 


@app.route('/', methods=['GET', 'POST'])
def index():
    input1 = None
    input2 = None
    similarity = None
    nli_prediction = None

    # Load saved model
    saved_model1 = BERT()
    saved_model1.load_state_dict(torch.load('models/trained-model1.pt'))

    if request.method == 'POST':
        # Clear the cache
        input1 = None
        input2 = None
        similarity = None
        nli_prediction  =None

        input1 = request.form['input1']
        input2 = request.form['input2']
        results = predict_nli_and_similarity(saved_model1, input1, input2, device='cpu')
        similarity= results["similarity_score"]
        nli_prediction = results["nli_label"]


        
    return render_template('index.html', input1=input1, input2=input2,similarity=similarity,nli_prediction=nli_prediction)

if __name__ == '__main__':
    app.run(debug=True)