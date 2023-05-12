
def basic_config():
    state = {}

    # original_addr = './sessionST'
    # state['train_addr'] = original_addr + '/session/'
    #
    # state['query_dict'] = '../statistic/dict/query_dict.pkl'
    # state['doc_dict'] = '../statistic/dict/doc_dict.pkl'

    # state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'
    # state['emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    # state['saveModeladdr'] = './model/'

    # state['train_rank_addr'] = original_addr + '/dataset/train/'
    # state['valid_rank_addr'] = original_addr + '/dataset/valid/'
    # state['test_rank_addr'] = original_addr + '/dataset/test/'

    # state['graph_type'] = 'query_click'
    # state['query_dict_addr'] = '../statistic/dict/qid2textid_dict.pkl'
    # state['doc_dict_addr'] = '../statistic/dict/did2textid_dict.pkl'
    #state['neib_dict_addr'] = './precess_data/neighbor_dictlist.pkl'# % state['graph_type']#query,click,query_doc_by_term
    #state['neib_freq_dict_addr'] = './precess_data/neighbor_freq_dictlist.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    
    #state['neib_dict_addr'] = './precess_data/neighbor_dictlist.fstop.loc0.pkl'# % state['graph_type']#query,click,query_doc_by_term
    #state['neib_freq_dict_addr'] = './precess_data/neighbor_freq_dictlist.fstop.loc0.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    #state['sample_neighbor_prefix'] = './precess_data/sample.neighbor.prefix1b{}.fstop.loc0.pkl'
    state['neib_dict_addr'] = './precess_data/neighbor_dictlist.pkl'# % state['graph_type']#query,click,query_doc_by_term
    state['neib_freq_dict_addr'] = './precess_data/neighbor_freq_dictlist.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    state['sample_neighbor_prefix'] = './precess_data/sample_neighbor_prefix_1b{}.pkl'
    #state['sample_neighbor_prefix'] = './precess_data/topsample.neighbor.5span.loc0.{}.pkl'
    
   
    state['neib_length_dict_addr'] = './precess_data/flat_length.pkl'
    state['neib_offset_dict_addr'] = './precess_data/flat_offset.pkl'
    state['BERT_folder'] = r'/data/shuting_wang/SubPer/bert'
    state['id2text_input'] = r'./precess_data/id2textid_input_ids.pkl'
    state['id2text_atten'] = r'./precess_data/id2textid_attention_mask.pkl'
    state['id2text_token'] = r'./precess_data/id2textid_token_type_ids.pkl'
    #state['bert_id2emb_dict'] = r'./precess_data/id2emb_contra_mypooler{}.pkl'
    state['bert_id2emb_dict'] = r'./precess_data/id2emb_contra{}_bt400.pkl'
    state['id2type_addr'] = r'./precess_data/id_type.pkl'
    state['bert_freeze_layer'] = 12
    # state['min_score_diff'] = 0.25
    # state['click_model'] = 'PSCM'

    # state['patience'] = 5
    # state['steps'] = 3000000  # Number of batches to process
    # state['pre_train_epoch'] = 2  # 5
    # state['train_freq'] = 50  # 200
    # state['eval_freq'] = 400  # 5000

    state['max_q_len'] = 30
    state['max_d_len'] = 20

    # state['aggregation_type'] = 'GraphSage'#GraphSage
    # state['ranker_optim'] = 'adam'  # adagrad,adam,adadelta
    # state['momentum'] = 0
    # state['ranker_lr'] = 0.001
    # state['weight_decay'] = 0

    # state['batch_size'] = 80
    # state['drate'] = 0.8
    # state['seed'] = 1234
    # state['clip_grad'] = 0.5

    state['embsize'] = 768
    state['edge_num'] = 8
    state['head_num'] = 6
    
    # state['text_dim'] = 50

    #state['ranker'] = 'ARCI' #ARCI, DSSM, LSTM-RNN
    # state['cost_threshold'] = 1.003
    # state['train_skg_flag'] = True

    # state['test_mode'] = False

    state['bert_pool'] = True
    state['bert_drt'] = 0.1
    state['graph_drt'] = 0.9
    
    state['n_heads'] = 6
    state['num_types'] = 2
    state['use_norm'] = 1
    state['max_session'] = 10
    state['hidden_dim'] = 64
    state['neig_sizes'] = '2,1,10,12,14'
    return state


def basic_config_tg():
    state = {}

    # original_addr = './sessionST'
    # state['train_addr'] = original_addr + '/session/'
    #
    # state['query_dict'] = '../statistic/dict/query_dict.pkl'
    # state['doc_dict'] = '../statistic/dict/doc_dict.pkl'

    # state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'
    # state['emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    # state['saveModeladdr'] = './model/'

    # state['train_rank_addr'] = original_addr + '/dataset/train/'
    # state['valid_rank_addr'] = original_addr + '/dataset/valid/'
    # state['test_rank_addr'] = original_addr + '/dataset/test/'

    # state['graph_type'] = 'query_click'
    # state['query_dict_addr'] = '../statistic/dict/qid2textid_dict.pkl'
    # state['doc_dict_addr'] = '../statistic/dict/did2textid_dict.pkl'
    
    state['neib_dict_addr'] = './process_data_tg/neighbor_dictlist.pkl'# % state['graph_type']#query,click,query_doc_by_term
    state['neib_freq_dict_addr'] = './process_data_tg/neighbor_freq_dictlist.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    state['sample_neighbor_prefix'] = './process_data_tg/sample_neighbor_prefix_1b{}.pkl'
    
    '''
    state['neib_dict_addr'] = './process_data_tg/neighbor_dictlist.degree2.pkl'# % state['graph_type']#query,click,query_doc_by_term
    state['neib_freq_dict_addr'] = './process_data_tg/neighbor_freq_dictlist.degree2.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    state['sample_neighbor_prefix'] = './process_data_tg/sample_neighbor_prefix_1b{}_degree2.pkl'
    '''
    #state['sample_neighbor_prefix'] = './process_data_tg/topsample.neighbor.loc0.{}.pkl'
    '''
    state['BERT_folder'] = r'BertChinese'
    state['bert_id2textid_dict'] = r'./process_data_tg/id2textid.pkl'
    state['id2text_input'] = r'./process_data_tg/id2textid_input_ids.pkl'
    state['id2text_atten'] = r'./process_data_tg/id2textid_attention_mask.pkl'
    state['id2text_token'] = r'./process_data_tg/id2textid_token_type_ids.pkl'
    '''
    
    state['BERT_folder'] = r'bert-base-chinese'
    state['bert_id2textid_dict'] = r'./process_data_tg/id2textid.base.pkl'
    state['id2text_input'] = r'./process_data_tg/id2textid_input_ids.base.pkl'
    state['id2text_atten'] = r'./process_data_tg/id2textid_attention_mask.base.pkl'
    state['id2text_token'] = r'./process_data_tg/id2textid_token_type_ids.base.pkl'
    
    
    
    #state['bert_id2emb_dict'] = r'./process_data_tg/id2emb_contra{}_base_bt100.pkl'
    state['bert_id2emb_dict'] = r'./process_data_tg/id2emb_contra{}_base_bt100.pkl'
    
    #state['bert_id2emb_dict'] = r'./process_data_tg/id2emb_last_contra{}_t0.01_base1pool_bt250.pkl'
    
    state['id2type_addr'] = r'./process_data_tg/id_type.base.pkl'
    # state['min_score_diff'] = 0.25
    # state['click_model'] = 'PSCM'

    # state['patience'] = 5
    # state['steps'] = 3000000  # Number of batches to process
    # state['pre_train_epoch'] = 2  # 5
    # state['train_freq'] = 50  # 200
    # state['eval_freq'] = 400  # 5000

    state['max_q_len'] = 10
    state['max_d_len'] = 25

    # state['aggregation_type'] = 'GraphSage'#GraphSage
    # state['ranker_optim'] = 'adam'  # adagrad,adam,adadelta
    # state['momentum'] = 0
    # state['ranker_lr'] = 0.001
    # state['weight_decay'] = 0

    # state['batch_size'] = 80
    # state['drate'] = 0.8
    # state['seed'] = 1234
    # state['clip_grad'] = 0.5

    state['embsize'] = 768
    state['edge_num'] = 8
    state['head_num'] = 6
    # state['text_dim'] = 50

    #state['ranker'] = 'ARCI' #ARCI, DSSM, LSTM-RNN
    # state['cost_threshold'] = 1.003
    # state['train_skg_flag'] = True

    # state['test_mode'] = False

    state['bert_pool'] = True
    state['bert_drt'] = 0.1
    state['graph_drt'] = 0.9
    
    state['n_heads'] = 6
    state['num_types'] = 2
    state['use_norm'] = 1
    state['max_session'] = 10
    
    state['hidden_dim'] = 768
    state['neig_sizes'] = '2,1,10,12,14'
    state['split_cnt'] = 9
    state['GraphSage'] = 'Aggregtor_HomoMax'
    
    return state

