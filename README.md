'''
To be continue....
'''

### File organization
```
HEXA
├── process_data_aol
├── process_data_tg
├── BertChinese
├── bert
├── aol
├── tiangong
├── output
│   ├── aol
│   ├── tiangong
├── model
├── *.py
```
--process_data_${dataset_name} #save the preprocessed graph datas

--output # save the output files containing the evaluating scores for validation/test samples, output files from datasets are saved in corresponding folders(output/${dataset_name}).  

--model #save the checkpoint of trained model.

--${dataset_name} #save the training/validation/test data for corresponding datasets.

--*.py #all used python files for experiments.

### Data Process
#### 1. Graph Construction

You can conduct graph construction by running the graph_allprocess_${dataset_name}.py. The correspoing data files will be saved in the folder, "process_data_${dataset_name}"

Note that the format original data used to generate training/validation/test samples and graph-related datas is json and each line represents a session. The format details are shown in below.
```
session = {
    "session_id": session_id, # str | session_id,
    "query": querys, # list | contains all search history of the session
}
querys[i] = {
    "clicks": clicks, # list | contains the i-th query's candidate documents' information.
    "text": text, # str | query's text content
    "query_id": str | query_id
}
clicks[i] = {
    "docid": docid, # str | the i-th candidate document's id
    "title": title, # str | the title of the i-th candidate document
    "label": label, # Tiangong-ST: 0-4 for the last query in the session & True or False for the previous queries | AOL: True or False 
}
```

We also provide the processed graph data by Google Drive 

For AOL: https://drive.google.com/file/d/1hs_4knfF_N_fN-eaZtg9rYjxurd2DZyr/view?usp=share_link

For Tiangong-ST: https://drive.google.com/file/d/1KiW6LVWjK8egpTc-WPn38rLPLE4zjUfe/view?usp=share_link

#### 2. Generation of training/validation/test data files.

In this step, you should process the original data to the format that is suitable to train the ranking model. 
You can write code to generate the train/validation/test data that conforms to the following format:
```
label \t q_1 \t d_1 \t ... \t q_K \t d_K \t q \t d \t gid_q_1 \t gid_d_1 \t ... \t gid_q_K \t gid_d_K \t gid_q \t gid_d
```
It means that there are K historical query (q_i) for the current query "q". Each history query has a clicked document (d_i). Based on this contextual information, we need to predict the relevance of the candidate "d". The relevance label of d is "label" (the first item).
gid_xxx denotes corresponding graph ids of these queries or documents, which are obtained by the step 1.


Note that you can obtain the Tiangong-ST dataset by this url "http://www.thuir.cn/tiangong-st/", and AOL dataset by contact to the author of the paper, "Context Attentive Document Ranking and Query Suggestion".


#### 3. Node Embedding Construction

Finally, we should prepare the initial node embeddings by training a two-tower Bert-based matching model. 

We provide the trained two-tower model by Google driver

For AOL: https://drive.google.com/file/d/1JE7K6hmcOLGkld14EDbymywZGGXlzFKB/view?usp=share_link

For Tiangong-ST: https://drive.google.com/file/d/1UW6Ln7212Q75FAZ4NDlUIdUWv74Vbrpk/view?usp=share_link

You can also train the two-tower model by the following command.
```
bash run_contra_tg.sh 
```

Then, run the following command to save the node embedding in the folder "process_data_${dataset_name}"
```
python get_id_embedding_${dataset_name}.py
```
We also provide the proprocessed node embeddings by Google Drive (Pleases refer to step 1).

### Training Model
After data processing, we can train the HEXA by following commands 
```
bash run_tg_cr.sh
```
We also provide the trained checkpoint in by google driver.

For AOL: https://drive.google.com/file/d/1gDdCDU_HFZhTUfQdVUIrBFoEyfixdLTl/view?usp=share_link

For Tiangong-ST: https://drive.google.com/file/d/1VVm3PL3UZE_Pj-qZIb4cS0Byih_w2qv_/view?usp=share_link

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{HEXA_2023WSDM,
author = {Wang, Shuting and Dou, Zhicheng and Zhu, Yutao},
title = {Heterogeneous Graph-Based Context-Aware Document Ranking},
year = {2023},
isbn = {9781450394079},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539597.3570390},
doi = {10.1145/3539597.3570390},
abstract = {Users' complex information needs usually require consecutive queries, which results in sessions with a series of interactions. Exploiting such contextual interactions has been proven to be favorable for result ranking. However, existing studies mainly model the contextual information independently and sequentially. They neglect the diverse information hidden in different relations and structured information of session elements as well as the valuable signals from other relevant sessions. In this paper, we propose HEXA, a heterogeneous graph-based context-aware document ranking framework. It exploits heterogeneous graphs to organize the contextual information and beneficial search logs for modeling user intents and ranking results. Specifically, we construct two heterogeneous graphs, i.e., a session graph and a query graph. The session graph is built from the current session queries and documents. Meanwhile, we sample the current query's k-layer neighbors from search logs to construct the query graph. Then, we employ heterogeneous graph neural networks and specialized readout functions on the two graphs to capture the user intents from local and global aspects. Finally, the document ranking scores are measured by how well the documents are matched with the two user intents. Results on two large-scale datasets confirm the effectiveness of our model.},
booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
pages = {724–732},
numpages = {9},
keywords = {context-aware document ranking, related sessions, heterogeneous graph},
location = {Singapore, Singapore},
series = {WSDM '23}
}
```
