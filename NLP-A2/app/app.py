from flask import Flask, render_template, request
import torch
import torchtext, math
import torch.nn as nn


app = Flask(__name__)

# Copy some neccessary code from A2.ipynb
# create model 
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size) # fc is the last layer for 
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #W_e
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #W_h
    
    def init_hidden(self, batch_size, device): 
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = None
    generation = None


    # Load the vocabulary from the saved file
    vocab = torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A2\vocab.pt')

    vocab_size = len(vocab)
    emb_dim = 1024                
    hid_dim = 50                
    num_layers = 1               
    dropout_rate = 0.5             
    lr = 1e-3
    device = 'cpu'

    # Load saved model
    loaded_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
    loaded_model.load_state_dict(torch.load(r'D:\AIT_lecture\NLP\code\Assignment\NLP-2025\NLP-A2\best-val-lstm_lm.pt'))
    loaded_model.eval() 

    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')    
    
    if request.method == 'POST':
        search_query = request.form['search_query']
        max_seq_len = 30
        seed = 0
        temperature = 1 # since we want the most make-sense sentence, temperature must highest which is 1

        generation = generate(search_query, max_seq_len, temperature, loaded_model, tokenizer, vocab, device, seed)
        generation = ' '.join(generation)
    return render_template('index.html', search_query=search_query,generation=generation)

if __name__ == '__main__':
    app.run(debug=True)