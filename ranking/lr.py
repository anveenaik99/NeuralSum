import os
import time
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from scipy.spatial.distance import cosine
import json
from gensim.models import Word2Vec
import pickle
import spacy

flags = tf.flags

flags.DEFINE_string ('data_dir',      'data',  'data directory, to compute vocab')
flags.DEFINE_string ('output_dir',    'cv',     'output directory, to store summaries')
flags.DEFINE_string ('nn_score_path_train', 'cv/scores_train',   'a json file storing sentence scores computed with neural model')
flags.DEFINE_string ('nn_score_path_test', 'cv/scores_test',   'a json file storing sentence scores computed with neural model')
flags.DEFINE_boolean('symbolic',       True,        'use symbolic features, e.g., sentence position, length')
flags.DEFINE_boolean('distributional', True,        'use distributional features, e.g., sentence position, length')
flags.DEFINE_string ('embedding_path', 'ranking/word2vec.model',      'emebdding path, which must be specified if distributional=True')
flags.DEFINE_integer('embedding_dim',  100,          'emebdding size')
flags.DEFINE_integer('max_doc_length',    500,   'max_doc_length')

FLAGS = flags.FLAGS
nlp = spacy.load('en_core_web_sm', entity=False)

def count_word(text):
    doc = nlp(text)
    tokens = [t.text for t in doc]
    tokens = [t for t in tokens if len(t.translate(t.maketrans('', '', string.punctuation + string.whitespace))) > 0] # + string.digits
    
    return len(tokens)

def load_wordvec(embedding_path):
    '''load word vectors'''

    print ('loading word vectors')
    # word_vec = {}
    # with open(embedding_path, "r") as f:
    #     for line in f:
    #         line = line.rstrip().split(' ')
    #         word_vec[line[0]] = np.asarray([float(x) for x in line[1:]])

    if os.path.exists(embedding_path):
        print("models exists")
        model = Word2Vec.load(embedding_path)
    else:
        print("Word2Vec Model do not exist. Terminating")
        exit()
    word_vec = {}
    for i, word in enumerate(model.wv.vocab):
        word_vec[word] = np.asarray([float(c) for c in model.wv[word]])
    print ('loading completed')
    return word_vec


def load_nn_score(nn_score_path):
    '''load the output scores predicted by an NN model
       this is a json file, which maps file name to a list of sentence scores'''
    scores = {}
    lines=np.loadtxt(nn_score_path,delimiter=' ')
    # print(type(lines))
    for i in range(len(lines)):
        # j=0
        # for j in range(1,len(lines[i])):
        #     if(lines[i][j]<1e-1 and lines[i][j-1]<1e-1):
        #         print(lines[i][j])
        #         break
        # print(j)
        # temp=lines[i][0:j]
        # scores[i]=temp
        # lines[i]=np.nan_to_num(lines[i])
        scores[i]=lines[i]
    # with open(nn_score_path, 'r') as f:
    #     lines=f.read
    #     for i in range(len(lines)):
    #         for key, val in line.iteritems():
    #             scores[key] = val
    # print(scores[1])
    return scores


def normalize(lx):
  '''normalize feature vectors in a small subset'''
  nsamples, nfeatures = len(lx), len(lx[0])
  for i in range(nfeatures):
    column = []
    for j in range(nsamples):
      column.append(lx[j][i])
    total = sum(column)
    for j in range(nsamples):
      if total!=0: lx[j][i] = lx[j][i] / total
  return lx


class Sybolic_Extractor(object):
    '''extract symbolic features: sentence length, position, entity counts
       We normalize all features.'''

    def __init__(self, etype='symbolic'):
        self.etype = etype
     
    @staticmethod 
    def length(sen):
        return len(sen)

    @staticmethod 
    def ent_count(sen):
        return sen.count('entity') 

    def extract_feature(self, sen_list):
        features = []
        for sid, sen in enumerate(sen_list):
            sen_feature = [sid, self.length(sen), self.ent_count(sen)]
            features.append(sen_feature) 

        return features


class Distributional_Extractor(object):
    '''extract distributional features: 
           sentence similary with respect to document
           sentence similary with respect to other sentences
       We normalize all features.'''

    def __init__(self, etype='distributional'):
        self.etype = etype

    @staticmethod 
    def compute_sen_vec(sen, word_vec):
        sen_vec = np.zeros(FLAGS.embedding_dim)
        count = 0
        for word in sen.split(' '):
            if word in word_vec:
                sen_vec += word_vec[word]
                count += 1
        if count > 0:
            sen_vec = sen_vec / count
       
        return sen_vec

    @staticmethod 
    def reduncy(sen_vec, doc_vec):
        return 1 - cosine(sen_vec, (doc_vec - sen_vec))

    @staticmethod 
    def relavence(sen_vec, doc_vec): 
        return 1 - cosine(sen_vec, doc_vec)

    def extract_feature(self, sen_list, word_vec):
        features = []
        sen_vec_list = []
        for sen in sen_list:
            sen_vec_list.append(self.compute_sen_vec(sen, word_vec))

        doc_vec = sum(sen_vec_list)       

        for sen_vec in sen_vec_list:
            sen_feature = [self.reduncy(sen_vec, doc_vec), self.relavence(sen_vec, doc_vec)]
            features.append(sen_feature)

        return features


def train_and_test():
    '''train and test a logistic regression classifier, which uses other features'''

    sExtractor = Sybolic_Extractor()
    dExtractor = Distributional_Extractor()

    word_vec = load_wordvec(FLAGS.embedding_path)

    nn_scores_train = load_nn_score(FLAGS.nn_score_path_train)
    nn_scores_test = load_nn_score(FLAGS.nn_score_path_test)

    train_x, train_y = [], []

    train_files = os.path.join(FLAGS.data_dir, 'test.json')
    with open(train_files) as f:
        examples = [json.loads(line) for line in f]
        lines=[]
        lines.append([])
        lines.append([])
        for segment in examples:
            temp_label=segment['labels']
            temp_label=temp_label.split('\n')
            temp_doc = segment['doc']
            temp_doc=temp_doc.split('\n')
            lines[0].append(temp_doc)
            lines[1].append(temp_label)
        for i in range(len(lines[0])):
            temp_doc=lines[0][i]
            temp_label=lines[1][i]
            sens = [sen.strip() for sen in temp_doc]
            y = [int(sen) for sen in temp_label]
            # print(len(sens),len(y)) 

            x_n = nn_scores_train[i]
            x_s = sExtractor.extract_feature(sens)
            x_d = dExtractor.extract_feature(sens, word_vec)
            x = [[f1] + f2 + f3 for f1, f2, f3 in zip(x_n, x_s, x_d)] 
            x = normalize(x)

            if len(y) > FLAGS.max_doc_length:
                y = y[:FLAGS.max_doc_length]

            train_x.extend(x)
            train_y.extend(y)

    f.close()

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_x=np.nan_to_num(train_x)
    train_y=np.nan_to_num(train_y)
    print(train_x.shape,train_y.shape)
    my_lr = lr()
    my_lr.fit(train_x, train_y)

    print ('testing...')
    output=[]
    num_sen=[]
    test_files = os.path.join(FLAGS.data_dir, 'test.json')
    with open(test_files) as f:
        examples = [json.loads(line) for line in f]
        lines=[]
        lines.append([])
        lines.append([])
        lines.append([])
        for segment in examples:
            temp_label=segment['labels']
            temp_label=temp_label.split('\n')
            temp_doc = segment['doc']
            temp_doc=temp_doc.split('\n')
            temp_summary = segment['summaries']
            temp_summary=temp_summary.split('\n')
            lines[0].append(temp_doc)
            lines[1].append(temp_label)
            lines[2].append(temp_summary)
        for i in range(len(lines[0])):
            temp_doc=lines[0][i]
            temp_label=lines[1][i]
            temp_summary = lines[2][i]
            sens = [sen.strip() for sen in temp_doc]
            y = [int(sen) for sen in temp_label]
            if len(y) > FLAGS.max_doc_length:
                y = y[:FLAGS.max_doc_length]
            if len(sens) > FLAGS.max_doc_length:
                sens = sens[:FLAGS.max_doc_length]
            # print(sens,y)

            x_n = nn_scores_test[i]
            x_s = sExtractor.extract_feature(sens)
            x_d = dExtractor.extract_feature(sens, word_vec)
            test_x = [[f1] + f2 + f3 for f1, f2, f3 in zip(x_n, x_s, x_d)] 
            test_x = normalize(test_x)

            f.close()
            test_x=np.asarray(test_x)
            test_x=np.nan_to_num(test_x)
            score = my_lr.predict_proba(test_x)
            # we need score for the postive classes
            sen_score = {}
            for sid, sentence in enumerate(sens):
                sen_score[sentence] = score[sid][1] + 0.5 * score[sid][2]

            sorted_sen = sorted(sen_score.items(), key=lambda d: d[1], reverse=True)  
            word_limit = count_word(temp_summary)
            summary_words = 0
            selected =[]
            for s in sorted_sen:
                temp_words = count_word(s[0])
                if (summary_words+temp_words)<word_limit:
                    summary_words+=temp_words
                    selected.append(s[0])
                else:
                    break

            # selected = [s[0] for s in sorted_sen[:3]]
            summary=[]
            count=0
            # store selected sentences to output file, following the original order
            for sen in sens:
                if sen in selected:
                    summary.append(sen)
                    count=count+1
            summary = ".".join(summary)
            output.append(summary)
            num_sen.append(count)
            print(count,len(selected))
            # if output is None:
            #     output = summary
            # else:
            #     output = np.vstack((output, summary)) 
            
    # file_name = '.'. 'test.output'
    num_sen=np.asarray(num_sen)
    print(num_sen.shape)
    print(np.mean(num_sen),np.std(num_sen))
    with open(os.path.join(FLAGS.output_dir, "summary"+str(time.time())+".txt"), 'w') as f:
        for line in output:
            f.write(line)
            f.write("\n")
        f.close
        # pickle.dump(output,f)
    # output_fp = open(os.path.join(FLAGS.output_dir, file_name), 'w')
    # for sen in sens:
    #     if sen in selected:
    #         output_fp.write(sen + '\n')
    # output_fp.close()

if __name__ == "__main__":
    train_and_test()

