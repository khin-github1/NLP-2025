# Importing libraries
from flask import Flask, render_template, request, jsonify
from gensim.models import KeyedVectors
from utils import Skipgram, SkipgramNeg, Glove
import pickle
import torch
import torch.nn.functional as F

# Importing training data
Data = pickle.load(open('./models/Data.pkl', 'rb'))
corpus = Data['corpus']
vocab = Data['vocab']
word2index = Data['word2index']
voc_size = Data['voc_size']
embed_size = Data['embedding_size']

# Load the models
model_save_path = './models/gensim_model.pkl'
gensim = pickle.load(open(model_save_path,'rb'))

# Instantiate the model and load saved parameters
skipgram = Skipgram(voc_size, embed_size)
skipgram.load_state_dict(torch.load('./models/Skipgram-v1.pt', map_location=torch.device('cpu')))
skipgram.eval()

# Instantiate the model and load saved parameters
skipgramNeg = SkipgramNeg(voc_size, embed_size)
skipgramNeg.load_state_dict(torch.load('./models/SkipgramNeg-v1.pt', map_location=torch.device('cpu')))
skipgramNeg.eval()

# Instantiate the model and load saved parameters
glove = Glove(voc_size, embed_size)
glove.load_state_dict(torch.load('./models/GloVe-v1.pt', map_location=torch.device('cpu')))
glove.eval()

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods = ['GET','POST'])
def index():
    # Home page
    if request.method == 'GET':
        return render_template('index.html', query='')
    
    # After user input
    if request.method == 'POST':
        query = request.form.get('query')
        results = [[]] # to store most similar words

        # Split the query into individual words
        query_words = query.split()

        vectors = [] # to store all input vectors
        result_vector = [] # to store final result
        for word in query_words:
            # Check if the word is in gensim model's vocab         
            if word in gensim:
                vectors.append(gensim.get_vector(word))
            else:
                vectors.append(gensim.get_vector('unknown'))

        # Linear addition of each vector of the input words to combine their meaning into one vector
        for i in range(len(vectors)):
            if i == 0:
                result_vector = vectors[i]
                print(result_vector)
            else:
                result_vector = result_vector + vectors[i]
        
        # Top 10 most similar
        search = gensim.most_similar(result_vector)
        for i in range(len(search)):
            results[0].append(search[i][0])

        # for other models
        models = [skipgram,skipgramNeg,glove]
        for i, model in enumerate(models):
            all_word_vectors = []
            for word in vocab:
                all_word_vectors.append(model.get_vector(word))
            all_word_vectors = torch.stack(all_word_vectors)

            vectors = [] # to store all input vectors
            result_vector = [] # to store final result
            for word in query_words:
                if word.lower() in vocab:
                    vectors.append(model.get_vector(word.lower()))
                else:
                    vectors.append(model.get_vector('<UNK>'))
            for i in range(len(vectors)):
                if i == 0:
                    result_vector = vectors[i]
                    print(result_vector)
                else:
                    result_vector = result_vector + vectors[i] # combine the embeddings of all input words

            # Calculate cosine similarities
            similarities = F.cosine_similarity(result_vector, all_word_vectors)

            # Get indices of the top ten similarities
            top_indices = torch.argsort(similarities, descending=True)[:10]

            # Fetch the corresponding words from the vocabulary
            top_words = [vocab[index.item()] for index in top_indices.view(-1)]

            # Append the top ten words to the results
            results.append(top_words[:10])

        
        models = ["GloVe(gensim)", "Skipgram", "SkipgramNeg", "GloVe"] # Table headings
        heading = "Most similar words are:" # to load only after submitting
        return render_template('index.html', query=query, heading=heading, models=models, results=results)

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)