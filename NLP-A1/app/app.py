
from flask import Flask, render_template, request
import torch
import json
from numpy import dot
from numpy.linalg import norm
import torch.nn as nn
import pickle
app = Flask(__name__)

from class_function import Skipgram, SkipgramNeg, Glove
# Importing training data
Data = pickle.load(open(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\Data.pkl', 'rb'))

corpus = Data['corpus']
vocabs = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['embedding_size']

from gensim.scripts.glove2word2vec import glove2word2vec  # Add this import
from gensim.models import KeyedVectors

# Load GloVe (Gensim) model
from gensim.test.utils import datapath
glove_file = datapath('glove.6B.100d.txt')  
word2vec_output_file = glove_file + '.word2vec'
glove2word2vec(glove_file, word2vec_output_file)  # Convert GloVe to Word2Vec format
gensim_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# GloVe module
class Glove(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(Glove, self).__init__()
        self.center_embedding = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        self.center_bias = nn.Embedding(voc_size, 1)
        self.outside_bias = nn.Embedding(voc_size, 1)

    def forward(self, center, outside, coocs, weighting):
        center_embeds = self.center_embedding(center)
        outside_embeds = self.outside_embedding(outside)
        center_bias = self.center_bias(center).squeeze(1)
        target_bias = self.outside_bias(outside).squeeze(1)
        inner_product = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
        return torch.sum(loss)

    def get_vector(self, word):
        if word not in word2index:
            raise KeyError(f"Word '{word}' not in vocabulary")
        id_tensor = torch.LongTensor([word2index[word]])
        v_embed = self.center_embedding(id_tensor)
        u_embed = self.outside_embedding(id_tensor)
        word_embed = (v_embed + u_embed) / 2
        return word_embed.squeeze(0)

# Cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Get top similar words for GloVe (Gensim)
def get_top_similar_words_gensim(model, word_input):
    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            # Get the most similar words using Gensim
            similar_words = model.most_similar(word_input, topn=10)
            return [f"{i+1}. {word} ({similarity:.8f})" for i, (word, similarity) in enumerate(similar_words)]
        else:
            return ["The system can search with 1 word only."]
    except KeyError:
        return ["The word is not in my corpus. Please enter a new word."]

# Get top similar words for custom models
def get_top_similar_words(model, word_input):
    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            word_embed = model.get_vector(word_input).detach().numpy().flatten()
            similarity_dict = {}
            for a in vocabs:
                a_embed = model.get_vector(a).detach().numpy().flatten()
                similarity_dict[a] = cos_sim(word_embed, a_embed)
            similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
            return [f"{i+1}. {similarity_dict_sorted[i][0]} ({similarity_dict_sorted[i][1]:.8f})" for i in range(10)]
        else:
            return ["The system can search with 1 word only."]
    except KeyError:
        return ["The word is not in my corpus. Please enter a new word."]

@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    glove_output = []
    gensim_output = []
    skipgram_output = []
    skipgram_neg_output = []

    if request.method == 'POST':
        search_query = request.form['search_query']

        # Load pre-trained models
        glove_model = Glove(voc_size, embed_size)
        glove_model.load_state_dict(torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\GloVe-v1.pt'))
        glove_model.eval()

        # Load Skipgram model
        state_dict = torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\skipgram_v1.pt')
        skipgram = Skipgram(voc_size, embed_size)
        skipgram.load_state_dict(state_dict, strict=False)
        skipgram.eval()

        # Load Skipgram with Negative Sampling model
        state_dict = torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\skipgramNeg_v1.pt')
        state_dict_remapped = {
            'embedding_center.weight': state_dict['center_embedding.weight'],
            'embedding_outside.weight': state_dict['outside_embedding.weight'],
        }
        skipgramNeg = SkipgramNeg(voc_size, embed_size)
        skipgramNeg.load_state_dict(state_dict_remapped)
        skipgramNeg.eval()

        # Get results for each model
        glove_output = get_top_similar_words(glove_model, search_query)
        skipgram_output = get_top_similar_words(skipgram, search_query)
        skipgram_neg_output = get_top_similar_words(skipgramNeg, search_query)
        gensim_output = get_top_similar_words_gensim(gensim_model, search_query)

    return render_template('index.html', search_query=search_query,
                           glove_output=glove_output, gensim_output=gensim_output,
                           skipgram_output=skipgram_output, skipgram_neg_output=skipgram_neg_output)

if __name__ == '__main__':
    app.run(debug=True)