
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
#gensim
# from gensim.test.utils import datapath
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('glove.6B.100d.txt')  
# gensim_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
# glove module
class Glove(nn.Module):
    def __init__(self, voc_size, emb_size):
        super(Glove, self).__init__()
        self.center_embedding  = nn.Embedding(voc_size, emb_size)
        self.outside_embedding = nn.Embedding(voc_size, emb_size)
        
        self.center_bias       = nn.Embedding(voc_size, 1) 
        self.outside_bias      = nn.Embedding(voc_size, 1)
    
    def forward(self, center, outside, coocs, weighting):
        center_embeds  = self.center_embedding(center) #(batch_size, 1, emb_size)
        outside_embeds = self.outside_embedding(outside) #(batch_size, 1, emb_size)
        
        center_bias    = self.center_bias(center).squeeze(1)
        target_bias    = self.outside_bias(outside).squeeze(1)
        
        inner_product  = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        #(batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1)
        
        # calculate loss
        loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
        
        return torch.sum(loss)
    
    def get_vector(self, word):
        """
        Get the embedding vector for a given word.
        :param word: The word to get the embedding for
        :return: The embedding vector as a PyTorch tensor
        """
        if word not in word2index:
            raise KeyError(f"Word '{word}' not in vocabulary")
        
        id_tensor = torch.LongTensor([word2index[word]])
        v_embed = self.center_embedding(id_tensor)
        u_embed = self.outside_embedding(id_tensor) 
        word_embed = (v_embed + u_embed) / 2 
        return word_embed.squeeze(0)  # Return as a PyTorch tensor (remove batch dimension)


def get_top_similar_words(model, word_input):
    """
    Get top 10 most similar words to the given word using the specified model.
    :param model: The word embedding model (e.g., Glove, Skipgram, SkipgramNeg)
    :param word_input: The input word to search for similar words
    :return: List of top 10 most similar words
    """
    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            # Get the word embedding of the input word
            word_embed = model.get_vector(word_input).detach().numpy().flatten()

            similarity_dict = {}

            # Compute cosine similarity for each word in the vocabulary
            for a in vocabs:
                a_embed = model.get_vector(a).detach().numpy().flatten()
                similarity_dict[a] = cos_sim(word_embed, a_embed)

            # Sort the dictionary by similarity in descending order
            similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

            # Return top 10 similar words
            return [f"{i+1}. {similarity_dict_sorted[i][0]} ({similarity_dict_sorted[i][1]:.4f})" for i in range(10)]

        else:
            return ["The system can search with 1 word only."]

    except KeyError:
        return ["The word is not in my corpus. Please enter a new word."]

# General function to get top 10 similar words
# Cosine similarity function
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


from fuzzywuzzy import process

def correct_spelling(word, vocab, threshold=80):
    """
    Correct the spelling of a word by finding the closest match in the vocabulary.
    :param word: The input word (potentially misspelled)
    :param vocab: The list of valid words in the vocabulary
    :param threshold: The minimum similarity score to accept a correction
    :return: The corrected word (or the original word if no match is found)
    """
    # Find the best match in the vocabulary
    match, score = process.extractOne(word, vocab)
    
    # Return the corrected word if the score is above the threshold
    if score >= threshold:
        return match
    else:
        return word  # Return the original word if no good match is found

def get_top_similar_words(model, word_input):
    """
    Get top 10 most similar words to the given word using the specified model.
    :param model: The word embedding model (e.g., Glove, Skipgram, SkipgramNeg)
    :param word_input: The input word to search for similar words
    :return: List of top 10 most similar words
    """
    try:
        if len(word_input.split()) == 1:  # Ensure input is a single word
            # Correct the spelling of the input word
            corrected_word = correct_spelling(word_input, vocabs)
            
            # If the corrected word is different, notify the user
            if corrected_word != word_input:
                print(f"Corrected '{word_input}' to '{corrected_word}'")
            
            # Get the word embedding of the corrected word
            word_embed = model.get_vector(corrected_word).detach().numpy().flatten()

            similarity_dict = {}

            # Compute cosine similarity for each word in the vocabulary
            for a in vocabs:
                a_embed = model.get_vector(a).detach().numpy().flatten()
                similarity_dict[a] = cos_sim(word_embed, a_embed)

            # Sort the dictionary by similarity in descending order
            similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)

            # Return top 10 similar words
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

        # # Load Gensim model (if applicable)
        # gensim_output = gensim_model.wv.most_similar(search_query, topn=10)
        # gensim_output = [f"{i+1}. {word} ({similarity:.4f})" for i, (word, similarity) in enumerate(gensim_output)]

        # List of models to loop through
        models = [("Glove", glove_model), ("Skipgram", skipgram), ("SkipgramNeg", skipgramNeg)]
        
        model_outputs = {}

        for model_name, model in models:
            try:
                model_outputs[model_name] = get_top_similar_words(model, search_query)
            except KeyError as e:
                model_outputs[model_name] = [str(e)]  # Handle unknown words
        
        # Assign outputs to respective variables
        glove_output = model_outputs.get("Glove", ["No results available."])
        skipgram_output = model_outputs.get("Skipgram", ["No results available."])
        skipgram_neg_output = model_outputs.get("SkipgramNeg", ["No results available."])

    return render_template('index.html', search_query=search_query, 
                           glove_output=glove_output, gensim_output=gensim_output, 
                           skipgram_output=skipgram_output, skipgram_neg_output=skipgram_neg_output)
if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request
# import torch
# import json
# from numpy import dot
# from numpy.linalg import norm
# import torch.nn as nn
# import pickle
# from gensim.models import Word2Vec

# app = Flask(__name__)

# # Importing training data
# Data = pickle.load(open(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\Data.pkl', 'rb'))

# corpus = Data['corpus']
# vocabs = Data['vocab']
# word2index = Data['word2index']
# voc_size = Data['voc_size']
# embed_size = Data['embedding_size']

# # Load Gensim model
# from gensim.test.utils import datapath
# from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec

# glove_file = datapath('glove.6B.100d.txt')  
# gensim = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# # Define models
# class Glove(nn.Module):
#     def __init__(self, voc_size, emb_size):
#         super(Glove, self).__init__()
#         self.center_embedding = nn.Embedding(voc_size, emb_size)
#         self.outside_embedding = nn.Embedding(voc_size, emb_size)
#         self.center_bias = nn.Embedding(voc_size, 1)
#         self.outside_bias = nn.Embedding(voc_size, 1)

#     def forward(self, center, outside, coocs, weighting):
#         center_embeds = self.center_embedding(center)  # (batch_size, 1, emb_size)
#         outside_embeds = self.outside_embedding(outside)  # (batch_size, 1, emb_size)
#         center_bias = self.center_bias(center).squeeze(1)
#         target_bias = self.outside_bias(outside).squeeze(1)
#         inner_product = outside_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
#         loss = weighting * torch.pow(inner_product + center_bias + target_bias - coocs, 2)
#         return torch.sum(loss)

#     # Corrected get_vector method for all models
#     def get_vector(self, word):
#         id_tensor = torch.LongTensor([word2index[word]])
#         v_embed = self.center_embedding(id_tensor)  # Use center_embedding, not embedding_center
#         u_embed = self.outside_embedding(id_tensor)  # Use outside_embedding, not embedding_outside
#         word_embed = (v_embed + u_embed) / 2  # Compute average embedding
    
#         # Detach tensor and convert to numpy for similarity calculation
#         word_embed = word_embed.detach().numpy().flatten()
#         return word_embed  # Return only the word embedding as a numpy array




# class Skipgram(nn.Module):
#     def __init__(self, voc_size, emb_size):
#         super(Skipgram,self).__init__()
#         self.embedding_center=nn.Embedding(voc_size,emb_size)
#         self.embedding_outside=nn.Embedding(voc_size,emb_size)
#     def forward(self,center,outside,all_vocabs):
#         center_embedding=self.embedding_center(center) #(batch_size, 1, emb_size)
#         outside_embedding=self.embedding_center(outside) #(batch_size, 1, emb_size)
#         all_vocabs_embedding=self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size

#         top_term=torch.exp(outside_embedding.bmm(center_embedding.transpose(1,2)).squeeze(2))  # bmm is dot product (ignore batch size) and reduce dim to 2
#         #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1)
        
#         lower_term=all_vocabs_embedding.bmm(center_embedding.transpose(1,2)).squeeze(2)
#         #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
#         lower_term_sum=torch.sum(torch.exp(lower_term),1) #(batch_size,1)
        
#         #calculate loss
#         loss=-torch.mean(torch.log(top_term/lower_term_sum))
        
#         return loss

#     def get_vector(self, word):
#         id_tensor = torch.LongTensor([word2index[word]])
#         v_embed = self.embedding_center(id_tensor)
#         u_embed = self.embedding_outside(id_tensor)
#         word_embed = (v_embed + u_embed) / 2  # Compute average embedding
    
#         # Detach tensor and convert to numpy for similarity calculation
#         word_embed = word_embed.detach().numpy().flatten()
#         return word_embed  # Return only the word embedding as a numpy array

# class SkipgramNeg(nn.Module):
    
#     def __init__(self, voc_size, emb_size):
#         super(SkipgramNeg, self).__init__()
#         self.embedding_center  = nn.Embedding(voc_size, emb_size)
#         self.embedding_outside = nn.Embedding(voc_size, emb_size)
#         self.logsigmoid        = nn.LogSigmoid()
    
#     def forward(self, center, outside, negative):
#         #center, outside:  (bs, 1)
#         #negative       :  (bs, k)
        
#         center_embed   = self.embedding_center(center) #(bs, 1, emb_size)
#         outside_embed  = self.embedding_outside(outside) #(bs, 1, emb_size)
#         negative_embed = self.embedding_outside(negative) #(bs, k, emb_size)
        
#         uovc           = outside_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, 1)
#         ukvc           = -negative_embed.bmm(center_embed.transpose(1, 2)).squeeze(2) #(bs, k)
#         ukvc_sum       = torch.sum(ukvc, 1).reshape(-1, 1) #(bs, 1)
        
#         loss           = self.logsigmoid(uovc) + self.logsigmoid(ukvc_sum)
        
#         return -torch.mean(loss)
#     def get_vector(self, word):
#         id_tensor = torch.LongTensor([word2index[word]])
#         v_embed = self.embedding_center(id_tensor)
#         u_embed = self.embedding_outside(id_tensor)
#         word_embed = (v_embed + u_embed) / 2  # Compute average embedding
    
#         # Detach tensor and convert to numpy for similarity calculation
#         word_embed = word_embed.detach().numpy().flatten()
#         return word_embed  # Return only the word embedding as a numpy array


# # Cosine similarity function
# def cos_sim(a, b):
#     return dot(a, b) / (norm(a) * norm(b))


# # General function to get top 10 similar words
# def get_top_similar_words(model, word_input):
#     try:
#         if len(word_input.split()) == 1:
#             word_embed = model.get_vector(word_input)  # Get the tensor directly
#             word_embed = word_embed.detach().numpy().flatten()  # Convert to numpy here
            
#             similarity_dict = {}
#             for a in vocabs:
#                 a_embed = model.get_vector(a)
#                 a_embed = a_embed.detach().numpy().flatten()  # Convert to numpy here
#                 similarity_dict[a] = cos_sim(word_embed, a_embed)

#             similarity_dict_sorted = sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
#             return [f"{i+1}. {similarity_dict_sorted[i][0]} ({similarity_dict_sorted[i][1]:.4f})" for i in range(10)]
#         else:
#             return ["The system can search with 1 word only."]
#     except KeyError:
#         return ["The word is not in my corpus. Please enter a new word."]


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     search_query = None
#     glove_output = None
#     gensim_output = None
#     skipgram_output = None
#     skipgram_neg_output = None

#     if request.method == 'POST':
#         search_query = request.form['search_query']

#         # Load pre-trained models
#         #glove
#         glove_model = Glove(voc_size, 2)
#         glove_model.load_state_dict(torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\GloVe-v1.pt'))
#         glove_model.eval()

#         #skipgram
#         state_dict = torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\skipgram_v1.pt')
#         skipgram = Skipgram(voc_size, embed_size)
#         skipgram.load_state_dict(state_dict,strict=False)
#         skipgram.eval()

#         #neg sampling
#         state_dict = torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A1\models\skipgramNeg_v1.pt')
#         state_dict_remapped = {
#         'embedding_center.weight': state_dict['center_embedding.weight'],
#         'embedding_outside.weight': state_dict['outside_embedding.weight'],
#         }

#         skipgramNeg = SkipgramNeg(voc_size, embed_size)
#         skipgramNeg.load_state_dict(state_dict_remapped)
#         skipgramNeg.eval()

#         # Generate outputs
#         glove_output = get_top_similar_words(glove_model, search_query)
#         gensim_output = gensim.most_similar(search_query, topn=10)
#         gensim_output = [f"{i+1}. {word} ({similarity:.4f})" for i, (word, similarity) in enumerate(gensim_output)]
#         skipgram_output = get_top_similar_words(skipgram, search_query)
#         skipgram_neg_output = get_top_similar_words(skipgramNeg, search_query)

#     return render_template('index.html', search_query=search_query, 
#                            glove_output=glove_output, gensim_output=gensim_output, 
#                            skipgram_output=skipgram_output, skipgram_neg_output=skipgram_neg_output)


# if __name__ == '__main__':
#     app.run(debug=True)
