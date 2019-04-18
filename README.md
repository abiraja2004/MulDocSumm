## Multi-document Summarization with CVAE
In progress

## Data
The data consist of 1) *rottentomatoes* moview reviews and 2) arguments from *Idebate*. Detailed description of the data is in [this paper by Wang et al.](https://www.aclweb.org/anthology/N16-1007)
The data can be downloaded in [Wang's page](http://www.ccs.neu.edu/home/luwang/) 
I used `prepare_data.py` to get a fixed number of documents per summary.
```
python prepare_data.py --file rottentomatoes --n 5 # 5 documents for a single summary
```
