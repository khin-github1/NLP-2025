# NLP-A2 LSTM Language Model for Four and Twenty Fairy Tales Text Generation

This project implements an LSTM-based language model to generate fairy tale text. The model is trained on a dataset of fairy tales, and it can generate coherent and contextually relevant text based on a given prompt. The project includes data preprocessing, model training, and text generation capabilities.

## Table of Contents
- [Student Information](#student-information)
- [Dataset](#dataset)
  - [Data Fetching](#data-fetching)
  - [Dataset Splits](#dataset-splits)
  - [Preprocessing Steps](#preprocessing-steps)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
  - [Training Results](#training-results)
- [Text Generation](#text-generation)
- [How to Run](#how-to-run)
- [Results](#results)

## Student Information
**Name:** Khin Yadanar Hlaing 

**ID:** st124959



## Dataset
The dataset used for this assignment is a collection of fairy tales stories (indcluding Cinderella, Sleeping Beauty and Beauty and the beat)fetched from [Project Gutenberg](https://www.gutenberg.org/). The text is preprocessed by tokenizing and splitting into sentences. The dataset is then divided into training, validation, and test sets.
### Data Fetching
The dataset is fetched using an HTTP GET request to the following URL:
```python
url = "https://www.gutenberg.org/cache/epub/52719/pg52719.txt"
```

The relevant content is extracted using specific markers in the text:
- **Start Marker:** "BLUE BEARD."
- **End Marker:** "in allusion to the story of Melusine."


### Dataset Splits
The dataset is split into three sets:
- **Training Set:** 6,319 sentences
- **Validation Set:** 1,114 sentences
- **Test Set:** 1,312 sentences

### Preprocessing Steps
1. **Tokenization:** The text is tokenized using the `basic_english` tokenizer from the `torchtext` library.
2. **Numericalization:** Words are converted into indices using a vocabulary built from the training set.
3. **Batching:** The data is split into batches for training and evaluation.

## Model Architecture
The LSTM language model consists of the following components:
- **Embedding Layer:** Converts word indices into dense vectors of fixed size.
- **LSTM Layer:** Processes the sequence of word embeddings and captures long-term dependencies.
- **Dropout Layer:** Applied after the embedding and LSTM layers to prevent overfitting.
- **Linear Layer:** Maps the LSTM output to the vocabulary size for word prediction.

### Hyperparameters
- **Vocabulary Size:** 9,547
- **Embedding Dimension:** 1,024
- **Hidden Dimension:** 50
- **Number of LSTM Layers:** 1
- **Dropout Rate:** 0.5
- **Learning Rate:** 1e-3

## Training Process
The model is trained using the Adam optimizer with a learning rate of 1e-3. The training process includes:

1. **Data Preprocessing:** Tokenizing and numericalizing the text data.
2. **Batch Processing:** Splitting the data into batches for training.
3. **Loss Function:** Cross-entropy loss is used to measure the model's performance.
4. **Training Loop:** The model is trained for 50 epochs, and the best model is saved based on validation loss.

### Training Results
- **Train Perplexity:** 95.084
- **Validation Perplexity:** 146.935
- **Test Perplexity:** 146.786

Lower perplexity scores indicate better performance, with the model making more accurate predictions on the given datasets.

## Text Generation
The trained model can generate text based on a given prompt. The process involves:
1. **Tokenization:** The input prompt is tokenized into words.
2. **Numericalization:** Words are converted into indices using the vocabulary.
3. **Prediction:** The model predicts the next word in the sequence based on the input.
4. **Temperature:** The temperature parameter controls the randomness of the generated text. Lower temperatures result in more deterministic outputs, while higher temperatures produce more diverse text.

## How to Run
1. **Install Dependencies:** Ensure you have the required libraries installed, including `torch`, `torchtext`, `nltk`, and `requests`.
2. **Run the Notebook:** Open the `LSTM-LM-fairytale.ipynb` notebook and run the cells to preprocess the data, train the model, and generate text.
3. To run app, 

 -Load the files from this repository

 -Run

```sh
python app.py
```
 -Access the app with http://127.0.0.1:5000 
## Results
[Download the Video](NLP-A2/video-language-model.mp4)

[Watch the Video](https://drive.google.com/drive/folders/1hyTL6PCqPajuEvy2jQVdmUuKsA1iFP6f)



