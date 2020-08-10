# NeuralSum
Neural Network Summarizer

## Features(to be included)
* **Hierarchical encoder**
..* **CNN sentence encoder**
..* **LSTM document encoder**
..* **Bidirectional LSTM document encoder**
* **Sentence extraction**
..* **Extraction with LSTM decoder**
..* **Prediction on top of BiLSTM encoder**
* **Word generator**
..* **Vanilla decoder**
..* **Hierarchical attention decoder**
..* **Beam-search decoder**
..* **External language model**
* **Post-process**
..* **LR Ranker**
..* **MERT feature-tuning**
..* **RL feature-tuning**

## Dependencies
* numpy
* scipy
* tensorflow
* scikit-learn

## Quick-start
* [Data](https://docs.google.com/uc?id=0B0Obe9L1qtsnSXZEd0JCenIyejg&export=download)
* You can change the data_dir in the code train.py, this directory should have three files, train.json, test.json and valid.json
* The preprocessing folder contains ```finding_similar_sentence.py``` this code generates the json files from general legal documents
* Pretrain a general-purpose encoder: ```python pretrain.py```
* Training ```python train.py```
* Evaluate ```python evaluate.py```, here you can set the load_model argument to specify the latest model that should be used to generate scores of the test data
* Run ```evaluate.py``` for train and test data to get scores for both the files. We need those to perform logistic regression to determine the probability of each sentence in the test data being in the final summary.
* Run ```python ranking/lr.py``` with the appropriate file names in place of scores_train and scores_test in arguments. 
* Run ```python find_rouge.py``` to finally calculate the rouge score of each generated summary as compared to the original summary. Change the filenames test_summary.txt and original_summary.txt appropriately

## Visualize scores
Sentence scores are stored during evaluation.

![score.png](./assets/score.png)


## Citation
```
@InProceedings{cheng-lapata:2016:P16-1, 
  author = {Cheng, Jianpeng and Lapata, Mirella}, 
  title = {Neural Summarization by Extracting Sentences and Words}, 
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)}, 
  year = {2016}, 
  address = {Berlin, Germany}, 
  publisher = {Association for Computational Linguistics}, 
  pages = {484--494} 
 }
```
## Reference
* [char-CNN-LSTM](https://github.com/carpedm20/lstm-char-cnn-tensorflow)
* [seq2seq](https://github.com/tensorflow/models/blob/master/textsum/seq2seq_attention_model.py)

## Liscense
MIT
