
def basic_config():
    state = {}
    
    state['neib_dict_addr'] = './process_data/neighbor_dictlist.pkl'# % state['graph_type']#query,click,query_doc_by_term
    state['neib_freq_dict_addr'] = './process_data/neighbor_freq_dictlist.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    state['sample_neighbor_prefix'] = './process_data/sample_neighbor_prefix_1b{}.pkl'
    
    state['neib_length_dict_addr'] = './process_data/flat_length.pkl'
    state['neib_offset_dict_addr'] = './process_data/flat_offset.pkl'
    state['BERT_folder'] = r'./bert'
    state['id2text_input'] = r'./process_data/id2textid_input_ids.pkl'
    state['id2text_atten'] = r'./process_data/id2textid_attention_mask.pkl'
    state['id2text_token'] = r'./process_data/id2textid_token_type_ids.pkl'
    state['bert_id2emb_dict'] = r'./process_data/id2emb_contra{}_bt400.pkl'
    state['id2type_addr'] = r'./process_data/id_type.pkl'
    state['bert_freeze_layer'] = 12

    state['max_q_len'] = 30
    state['max_d_len'] = 20

    state['embsize'] = 768
    state['edge_num'] = 8

    state['bert_pool'] = True
    state['bert_drt'] = 0.1
    state['graph_drt'] = 0.9
    
    state['n_heads'] = 6
    state['num_types'] = 2
    state['use_norm'] = 1
    state['max_session'] = 10
    
    state['hidden_dim'] = 64
    
    state['GraphSage'] = 'Aggregtor_HomoMean'
    state['split_cnt'] = 9
    return state


def basic_config_tg():
    state = {}

    
    state['neib_dict_addr'] = './process_data_tg/neighbor_dictlist.pkl'# % state['graph_type']#query,click,query_doc_by_term
    state['neib_freq_dict_addr'] = './process_data_tg/neighbor_freq_dictlist.pkl'  # % state['graph_type']#query,click,query_doc_by_term
    state['sample_neighbor_prefix'] = './process_data_tg/sample_neighbor_prefix_1b{}.pkl'
    
    state['BERT_folder'] = r'bert-base-chinese'
    state['bert_id2textid_dict'] = r'./process_data_tg/id2textid.base.pkl'
    state['id2text_input'] = r'./process_data_tg/id2textid_input_ids.base.pkl'
    state['id2text_atten'] = r'./process_data_tg/id2textid_attention_mask.base.pkl'
    state['id2text_token'] = r'./process_data_tg/id2textid_token_type_ids.base.pkl'
    
    state['bert_id2emb_dict'] = r'./process_data_tg/id2emb_contra{}_base_bt100.pkl'
    
    state['id2type_addr'] = r'./process_data_tg/id_type.base.pkl'

    state['max_q_len'] = 10
    state['max_d_len'] = 25

    state['embsize'] = 768
    state['edge_num'] = 8
    state['head_num'] = 6

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

