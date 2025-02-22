#  A5: Do you AGREE?

- [Student Information](#student-information)
- [Installation and Setup](#installation-steps)
- [Task 1 - Training BERT from Scratch](#task-1---training-bert-from-scratch)
- [Task 2 - Sentence BERT](#task-2---sentence-bert)
- [Task 3 - Evaluation and Analysis](#task-3---evaluation-and-analysis)
- [Task 4 - Web Application](#task-4---web-application)
    - [Web Page Result](#result)
    - [Usage](#usage)

## Student Information
 - Name: Khin Yadanar Hlaing
 - ID: st124959

## Installation Steps
To run app, 
1. Load the files from this repository
2. Run
```sh
python app.py
```
3. Access the app with http://127.0.0.1:5000 

## Task 1 - Training BERT from Scratch

- Dataset: [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) 
- The hyperparameters chosen for training our BERT model was:  
    - Number of Encoder of Encoder Layer - 6  
    - Number of heads in Multi-Head Attention - 8  
    - Embedding Size/ Hidden Dim - 768  
    - Number of epochs - 1000  
    - Training data - 740042 sentences
    - dimension of K(=Q), V  =64
    - Vocab size - 60305  
- Save Model: Then, I save model weights which can be located in the `app/models/bert-from-scratch.pt`


## Task 2 - Sentence BERT
- Dataset: [MNLI](https://huggingface.co/datasets/glue/viewer/mnli)
- The hyperparameters chosen for tuning S-BERT on pretrained  BERT model was:
    - Training data - 10000 rows  
    - Number of epochs - 2 
- The hyperparameters chosen for tuning S-BERT on our BERT model was:
    - Training data - 10000 rows  
    - Number of epochs - 5  

- I trained two models for sentence BERT: using pretrained BERT and  trained BERT from task 1 previously save as our-model.pt.


## Task 3 - Evaluation and Analysis

| Model Type | trainable parameter | Average Cosine Similarity | Average Loss | Cosine Similarity with one specific pair sentence  | Accuracy| Precision | Recall,F1-score | Training Time (train with MNLI dataset)
|:--------------------------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---------------------:|
| S-BERT on our BERT model        |    83,137,171   |    0.9986     |  8.9522 | 0.9996 |  0.3190| 0.1018 | 0.3190,0.1543|  564m 2s(on CPU)    |
| S-BERT(Pre)           |   109,482,240    |    0.7733 |  1.0790 | 0.8057 | 0.3410 | - | - | 139m 54s(num-epoch=2)       |  


## Task 4 - Web Application

### Web Page Result
![Contradiction](images/image.png)
![Entailment](images/entailment.png)

### Usage:
- Input: There are two input fields where you can enter two sentences to compare.
- Output: after you click the 'Analyze Similarity' button, the NLI Classification can be seen.