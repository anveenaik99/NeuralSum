from rouge import FilesRouge
from data_reader import original_summary
import os

data_dir="data"
output_dir="cv"
filename="test.json"
original_file=os.path.join(output_dir,"original_summary.txt")
if(not os.path.exists(original_file)):
	original_summary(os.path.join(data_dir,filename),output_dir)
print(original_file)
files_rouge = FilesRouge()
scores = files_rouge.get_scores(os.path.join(output_dir,"test_summary.txt"),original_file)
# scores = files_rouge.get_scores(str(original_file), str(original_file))
# or
# scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
print(scores)