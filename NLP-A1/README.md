# NLP_A1_Engine_Search




## Student Information
Name - Khin Yadanar Hlaing 
ID - st124959


To run app, 
1. Load the files from this repository
2. Run
```sh
python app/app.py
```
3. Access the app with http://127.0.0.1:5000  (but the app has not finished yet)
## Usage
Enter Input a single word on a search bar  and display the top 10 most similar words from each model's vocabulary.

## Training Data
Corpus source - nltk datasets('abc') : Austrailian Broadcasting Commission  
Token Count |C| - 241109  
Vocabulary Size |V| - 14270
Embedding dimension - 2  
Learning rate - 0.001  
Epochs - 1000  

Training parameters are consistant across all three models.  

## Word Embedding Models Comparison

| Model             | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
|-------------------|-------------|---------------|---------------|--------------------|-------------------|
| Skipgram          | 2     | 9.68      | 9 min 42 sec       | 0.00%            | 0.00%           |
| Skipgram (NEG)    | 2     | 3.11       | 10 min 30 sec       | 0.00%            | 0.00%           |
| Glove             | 2     | 0.42       | 2 min 27 sec       | 0.00%            | 0.00%           |
| Glove (Gensim)    | -     | -       | -       | 55.45%            | 93.87%           |

## Similarity Scores

| Model               | Skipgram | Skipgram (NEG) | GloVe | GloVe (Gensim) | Y true |
|---------------------|-----------|----------------|-------|----------------|--------|
| **Spearman Correlation**             | 0.192   | 0.128        | 0.26 | 0.6035        | 0.8735 |


## Model Comparison Report

