import numpy
import gensim
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET 
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tqdm import tqdm
import pickle
import os
import string
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
from spacy.attrs import ORTH
import time
import json

# nltk.set_proxy('http://172.16.2.30:8080')
# nltk.download('popular')

raw_dir = "./raw_dir"
os.makedirs(raw_dir, exist_ok = True)
summary_path = os.path.join(raw_dir, "summary.pkl")
full_path = os.path.join(raw_dir, "full.pkl")
data_dir="test_data"
full_txt="Full-and-Summary-Docs/Full-Docs/full-txt"
# full_txt="train-docs"
summ_txt="Full-and-Summary-Docs/Summary/A1/full txt"
# summ_txt="summary"
save_output="train_NeuralSum"+str(time.time())+".json"

def custom_sentencizer(doc):
    ''' Look for sentence start tokens by scanning for periods only. '''
    for i, token in enumerate(doc[:-2]):  # The last token cannot start a sentence
        if token.text == ".":
            #doc[i+1].is_sent_start = True
            pass
        else:
            doc[i+1].is_sent_start = False  # Tell the default sentencizer to ignore this token
    return doc

def store(path, obj):
	with open(path, 'wb') as f_out:
		pickle.dump(obj, f_out)

if os.path.exists(summary_path):
	print("summary and full exists")
	summary = pickle.load(open(summary_path, 'rb'))
	full = pickle.load(open(full_path, 'rb'))
else:
	print("summary and full do not exists")
	summary = list()
	full = list()
	pname=os.path.join(data_dir,full_txt)
	for fname in tqdm(os.listdir(pname)):
		print(fname)
		with open(os.path.join(pname,fname)) as f:
			a=f.read().strip()
			a = a.replace('\n', ' ')
			a = re.sub(' +', ' ', a)
			# print(os.path.join(pname,fname))
			full.append(a)
	pname=os.path.join(data_dir,summ_txt)
	for fname in os.listdir(pname):
		print(os.path.join(pname,fname))
		with open(os.path.join(pname,fname)) as f:
			a=f.read().strip()
			a = a.replace('\n', ' ')
			a = re.sub(' +', ' ', a)
			# print(os.path.join(pname,fname))
			summary.append(a)
	# l=pickle.load(open("order.pkl",'rb'))
	# for ele in range(7001,7132):
	# 	fname=l[ele]
	# 	print(fname)
	# 	with open(fname) as f:
	# 		a=f.read().strip()
	# 		a = a.replace('\n', ' ')
	# 		a = re.sub(' +', ' ', a)
	# 		# print(len(a))
	# 		full.append(a)
	# for ele in range(14133,len(l)):
	# 	fname=l[ele]
	# 	print(fname)
	# 	with open(fname) as f:
	# 		a=f.read().strip()
	# 		a = a.replace('\n', ' ')
	# 		a = re.sub(' +', ' ', a)
	# 		summary.append(a)
	store(summary_path, summary)
	store(full_path, full)


print(len(summary))
print(len(full))

texts = list()
for i in summary:
	texts.append(i)
for i in full:
	texts.append(i)
texts = list(filter(None, texts))

process_dir = "./process"
os.makedirs(process_dir, exist_ok=True)
processed_data_path = os.path.join(process_dir, "process.pkl") 
dictionary_path = os.path.join(process_dir, "dict.pkl")
corpus_path = os.path.join(process_dir, "corpus.pkl")

model_dir = "./model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "tfidf.model")

if os.path.exists(processed_data_path):
	print("processed_data exists")
	processed_data = pickle.load(open(processed_data_path, 'rb'))
else:
	print("processed_data do not exists")
	processed_data = [gensim.utils.simple_preprocess(x) for x in tqdm(texts)]
	store(processed_data_path, processed_data)

if os.path.exists(dictionary_path):
	print("dictionary exists")
	dct = pickle.load(open(dictionary_path, 'rb'))
else:
	print("dictionary do not exists")
	dct = Dictionary(processed_data)  # fit dictionary
	store(dictionary_path, dct)

if os.path.exists(corpus_path):
	print("corpus exists")
	corpus = pickle.load(open(corpus_path, 'rb'))
else:
	print("corpus do not exists")
	corpus = [dct.doc2bow(line) for line in processed_data]  # convert corpus to BoW format
	store(corpus_path, corpus)
if os.path.exists(model_path):
	print("models exists")
	model = TfidfModel.load(model_path)
else:
	print("model do not exists")
	model = TfidfModel(corpus)  # fit model
	model.save(model_path)

# vector = model[corpus[1]]

def find_vector(para1, para2):
	for l, i in enumerate(para1):
		data = gensim.utils.simple_preprocess(i)
		corp = dct.doc2bow(data)
		vector = model[corp]
		vector_comp = np.zeros(len(dct))
		for j in vector:
			vector_comp[j[0]] = j[1]

		vector_comp = vector_comp.reshape(1,-1)
		if(l==0):
			final_array = vector_comp
			# print("hi")
		else:
			final_array = np.append(final_array, vector_comp, axis = 0)
			# print("yo")
	for l, i in enumerate(para2):
		data = gensim.utils.simple_preprocess(i)
		corp = dct.doc2bow(data)
		vector = model[corp]
		vector_comp = np.zeros(len(dct))
		for j in vector:
			vector_comp[j[0]] = j[1]

		vector_comp = vector_comp.reshape(1,-1)
		final_array = np.append(final_array, vector_comp, axis = 0)

	val = min(150, final_array.shape[0])
	pca = PCA(n_components = val)
	processed_vector = pca.fit_transform(final_array)
	return processed_vector

def check(para, prob_candidate):
	
	act = open("acts_in_docs.txt", "r")
	acts = list()
	for x in act:
		acts.append(x.replace('\n', ''))

	legal_word = open("dict_words.txt", "r")
	legal_words = list()
	for x in legal_word:
		legal_words.append(x.replace('\n', ''))
	result = list()
	threshold = 0.2
	for j, i in enumerate(prob_candidate):
		temp = para[j] #########convert
		temp=temp.lower()
		line = temp.translate(temp.maketrans("","", string.punctuation))
		line=re.sub(' +',' ',line)
		count_act = 0
		for x in acts:
			if x in line:
				count_act += 1
		count_word = 0
		for x in legal_words:
			if x in line:
				count_word += 1
		metric = float(1.0*(count_word+count_act))/float(len(line.split(" ")))
		# print(count_act, count_word, metric)
		if metric>threshold:
			result.append(i)

	return result
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(custom_sentencizer, before = "parser")

special_cases = {"Rs.": "Rs.", "No.": "No.", "no.": "No.", "vs.": "vs", "i.e.": "i.e.", "viz.": "viz.", "M/s.": "M/s.", "Mohd.": "Mohd.", "Ex.": "exhibit", "Art." : "article", "Arts." : "articles", "S.": "section", "s.": "section", "ss.": "sections", "u/s.": "section", "u/ss.": "sections", "art.": "article", "arts.": "articles", "u/arts." : "articles", "u/art." : "article"}

for case, orth in special_cases.items():
	nlp.tokenizer.add_special_case(case, [{ORTH: orth}])

label_path = os.path.join(process_dir, "labels.pkl")
save_vector = list()
examples=[]
for i in range(len(summary)):
	# split para in sentences using tokenizer
	# para_1 = sent_tokenize(summary[i])
	# para_2 = sent_tokenize(full[i])
	if summary[i] is None:
		continue
	p_1 = nlp(summary[i])########list of sentences
	p_2 = nlp(full[i])
	para_1=[]
	para_2=[]
	for s in p_1.sents:
		para_1.append(str(s))
	for r in p_2.sents:
		para_2.append(str(r)) #sentences in full text
	label = np.zeros(len(para_2))
	if(len(para_2)==0 or len(para_1)==0):##########remove this
		continue
	#compute the vector for each sentence
	processed_vector = find_vector(para_1, para_2)
	summary_vector = processed_vector[:len(para_1),:]
	full_vector = processed_vector[len(para_1):, :]
	#compute similarity between vectors
	similarity = cosine_similarity(summary_vector, full_vector)
	sort_indices = np.argsort(similarity, axis = 1)
	#making most important as 1
	#shape 1, (no. of sentences in para_2)
	label[sort_indices[:,-1]] = 2
	#Take top 5%
	top_val = int(0.05*sort_indices.shape[1])
	prob_candidate = np.unique(sort_indices[:, -top_val:-1].reshape(1, -1))
	#check if the any of prob candidate should be labeled as may be
	maybe_vector = check(para_2, prob_candidate)
	label[maybe_vector] = 1
	# print(label.shape, prob_candidate.shape, len(maybe_vector))
	save_vector.append(label.reshape(1, -1))

	summaries=[]
	labels=[]
	for j in range(len(label)):
		if label[j]==2:
			summaries.append(para_2[j])
			labels.append('2')
		elif label[j] == 1:
			labels.append('1')
		else:
			labels.append('0')
	print(len(summaries),len(para_2),len(labels))
	ex = {'doc':'\n'.join(para_2),'labels':'\n'.join(labels),'summaries':'\n'.join(summaries)}
	# print(ex)
	print(i, len(summary))
	examples.append(ex)
	if(i%1 == 10):
		store(label_path, save_vector)

		with open(save_output,'w') as f:
			for row in examples:
				f.write(json.dumps(row, ensure_ascii=False) + "\n")
		f.close()
	# if(i%5 == 0):
	# 	store(label_path, save_vector)
	# print(len(save_vector), save_vector[i].shape)
	