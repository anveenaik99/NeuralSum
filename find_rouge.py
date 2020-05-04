from rouge import FilesRouge, Rouge
from data_reader import original_summary
import os
import json
import numpy as np

data_dir="data"
output_dir="cv"
rouge_filename="rouge_scores.json"
filename="test.json"
original_file=os.path.join(output_dir,"original_summary.txt")
if(not os.path.exists(original_file)):
	original_summary(os.path.join(data_dir,filename),output_dir)
# print(original_file)
# files_rouge = FilesRouge()
# scores = files_rouge.get_scores(os.path.join(output_dir,"test_summary.txt"),os.path.join(output_dir,"original_summary.txt"))
# scores = files_rouge.get_scores(os.path.join(output_dir,"test_summary.txt"), os.path.join(output_dir,"test_summary.txt"))
# or
# scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
# hyp=np.loadtxt()
# ref=np.loadtxt()

hyp=[]
ref=[]
with open(original_file) as f:
    hyp = f.readlines()
with open(os.path.join(output_dir,"test_summary.txt")) as fp:
    ref = fp.readlines()
rouge = Rouge()
scores = rouge.get_scores(hyp, ref)

print(len(scores))
with open(os.path.join(output_dir,rouge_filename),'w') as f:
	for row in scores:
		f.write(json.dumps(row, ensure_ascii=False) + "\n")
f.close()