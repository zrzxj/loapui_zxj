#coding=utf-8
'''
Created on Dec 25, 2016

@author: loapui
'''
from os.path import join

import pandas as pd

#from featureEngineer.base import train_path, test_path, feature_path
#from featureEngineer.features import get_card_feature, get_score_feature
from base import train_path, test_path, feature_path
from features import get_card_feature, get_score_feature

def _rank(feature):
    distinct_val = set(feature)
    sorted_distinct_val = sorted(distinct_val, key=lambda x: x)
    ranking_feature = [list(sorted_distinct_val).index(x)+1 for x in feature.tolist()]
    return ranking_feature


def transform_ranking_feature():
    id_train = pd.read_csv(join(train_path, 'subsidy_train.txt'), sep=',', header=None)
    id_train.columns = ['id', 'money']
    id_train = id_train[['id']]
    id_test = pd.read_csv(join(test_path, 'studentID_test.txt'), sep=',', header=None)
    id_test.columns = ['id']
    
    all_feature = pd.concat([id_train, id_test])
    card_feature_train = get_card_feature("train")
    card_feature_test = get_card_feature("test")
    card_feature = pd.concat([card_feature_train, card_feature_test])
    all_feature = pd.merge(all_feature, card_feature, how='left', on='id')
    score_feature = get_score_feature()
    all_feature = pd.merge(all_feature, score_feature, how='left', on='id')
    
    #fill na
    all_feature = all_feature.fillna(0)
    
    all_ranking_feature = pd.concat([id_train, id_test])
    for _feat in all_feature.columns:
        if _feat == 'id':
            continue
        rank_feat = _rank(all_feature[_feat])
        new_feat = _feat.upper() + '_RANKING'
        all_ranking_feature[new_feat] = rank_feat
    
    # print all_ranking_feature.head()
    all_ranking_feature.to_csv(join(feature_path, 'RANKING_features.csv'), index=False)
    
    
def extract_id_feature():
    id_train = pd.read_csv(join(train_path, 'subsidy_train.txt'), sep=',', header=None)
    id_train.columns = ['id', 'money']
    id_train = id_train[['id']]
    id_test = pd.read_csv(join(test_path, 'studentID_test.txt'), sep=',', header=None)
    id_test.columns = ['id']
    
    id_train_test = pd.concat([id_train, id_test])
    
    nbin_scale = [3, 6, 12, 24, 36, 72, 144, 288, 512, 1024, 10000]
    
    for i, scale in enumerate(nbin_scale):
        cat_i = pd.cut(id_train_test[['id']], scale)
        feat_name = 'id_' + str(i)
        id_train_test[feat_name] = cat_i.codes
        
    id_train_test.to_csv(join(feature_path, 'id_features.csv'), index=False)
    
    
def extract_score_feature():
    score_train = pd.read_csv(join(train_path, 'score_train.txt'), sep=',', header=None)
    score_train.columns = ['id', 'college', 'score']
    score_test = pd.read_csv(join(test_path, 'score_test.txt'), sep=',', header=None)
    score_test.columns = ['id', 'college', 'score']
    score_train_test = pd.concat([score_train, score_test])
    score_train_test = score_train_test.drop_duplicates()
    
    
    college = pd.DataFrame(score_train_test.groupby(['college'])['score'].max())
    college = college.reset_index()
    college.columns = ['college', 'num']
    
    score_train_test = pd.merge(score_train_test, college, how='left', on='college')
    score_train_test['order'] = score_train_test['score'] / score_train_test['num']
    score_train_test.columns = ['id', 'academy', 'score_rank', 'nStudent_of_academy', 'rate_rank']
    
    score_train_test.to_csv(join(feature_path, 'score_features.csv'), index=False)


def extract_card_feature(flag):
    card_file = {'train' : join(train_path, 'card_train.txt'),
             'test' : join(test_path, 'card_test.txt')}
    
    card_records = pd.read_csv(card_file[flag], sep=',', header=None)
    card_records.columns = ['id', 'consume', 'where', 'how', 'time', 'amount', 'remainder']
    # pre process
    card_records = card_records.drop_duplicates()
    # card_records = card_records.dropna()
    card_records['amount'] = card_records['amount'].abs()
    card_records['diff'] = card_records['remainder'] - card_records['amount']
    
    # pos type feats
    print('  -> pos type feats')
    card = pd.DataFrame(card_records.groupby(['id', 'consume', 'how'])['amount'].max())
    card.columns = ['amount_max']
    card['amount_min'] = card_records.groupby(['id', 'consume', 'how'])['amount'].min()
    card['amount_avg'] = card_records.groupby(['id', 'consume', 'how'])['amount'].mean()
    card['amount_sum'] = card_records.groupby(['id', 'consume', 'how'])['amount'].sum()
    card['amount_num'] = card_records.groupby(['id', 'consume', 'how'])['amount'].count()
    card = card.reset_index()
    how_list = ["食堂", "超市", "开水", "图书馆", "洗衣房", "文印中心", "淋浴", "教务处", "校车", "校医院", "其他"]
    card = card[card['how'].isin(how_list)]
    card_feature = pd.DataFrame(card['id'].unique(),columns=['id'])
    for _type in how_list:
        pos_type = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][(card.consume=='POS消费')&(card.how==_type)]
        pos_type.columns = ['id', 'max_'+_type+'_pos',\
                            'min_'+_type+'_pos', 'avg_'+_type+'_pos',\
                            'sum_'+_type+'_pos', 'num_'+_type+'_pos']
        card_feature = pd.merge(card_feature, pos_type, how='left', on='id')
    del pos_type
        
    card = pd.DataFrame(card_records.groupby(['id', 'consume'])['amount'].max())
    card.columns = ['amount_max']
    card['amount_min'] = card_records.groupby(['id', 'consume'])['amount'].min()
    card['amount_avg'] = card_records.groupby(['id', 'consume'])['amount'].mean()
    card['amount_sum'] = card_records.groupby(['id', 'consume'])['amount'].sum()
    card['amount_num'] = card_records.groupby(['id', 'consume'])['amount'].count()
    card = card.reset_index()
    
    # pos all feats
    print('  -> pos all feats')
    pos_all = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='POS消费']
    pos_all.columns = ['id', 'max_all_pos', 'min_all_pos', 'num_all_pos', 'sum_all_pos', 'avg_all_pos']
    card_feature = pd.merge(card_feature, pos_all, how='left', on='id')
    del pos_all
    
    # transfer feats
    print('  -> transfer feats')
    transfer = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='圈存转账']
    transfer.columns = ['id', 'max_transfer', 'min_transfer', 'avg_transfer', 'sum_transfer', 'num_transfer']
    card_feature = pd.merge(card_feature, transfer, how='left', on='id')
    del transfer
    
    # payget feats
    print('  -> payget feats')
    payget = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='支付领取']
    payget.columns = ['id', 'max_payget', 'min_payget', 'avg_payget', 'sum_payget', 'num_payget']
    card_feature = pd.merge(card_feature, payget, how='left', on='id')
    del payget
    
    # card open
    print('  -> card open feats')
    card_open = card[['id', 'amount_num']][card.consume=='卡片开户']
    card_open.columns = ['id', 'num_of_open_card']
    card_feature = pd.merge(card_feature, card_open, how='left', on='id')
    del card_open
    
    # card close
    print('  -> card close feats')
    card_close = card[['id', 'amount_num']][card.consume=='卡片销户']
    card_close.columns = ['id', 'num_of_close_card']
    card_feature = pd.merge(card_feature, card_close, how='left', on='id')
    del card_close
    
    # card lost
    print('  -> card lost feats')
    card_lost = card[['id', 'amount_num']][card.consume=='卡挂失']
    card_lost.columns = ['id', 'num_of_lost_card']
    card_feature = pd.merge(card_feature, card_lost, how='left', on='id')
    del card_lost
    
    # recharge
    print('  -> recharge feats')
    recharge_amount = card[['id', 'amount_max', 'amount_min', 'amount_avg', 'amount_sum', 'amount_num']][card.consume=='卡充值']
    recharge_amount.columns = ['id', "max_account_recharge", "min_account_recharge", "avg_account_recharge", "sum_account_recharge", "num_account_recharge"]
    card_feature = pd.merge(card_feature, recharge_amount, how='left', on='id')
    card = pd.DataFrame(card_records.groupby(['id', 'consume'])['diff'].max())
    card.columns = ['diff_max']
    card['diff_min'] = card_records.groupby(['id', 'consume'])['diff'].min()
    card['diff_avg'] = card_records.groupby(['id', 'consume'])['diff'].mean()
    card['diff_sum'] = card_records.groupby(['id', 'consume'])['diff'].sum()
    card = card.reset_index()
    recharge_diff = card[['id', 'diff_max', 'diff_min', 'diff_avg', 'diff_sum']][card.consume=='卡充值']
    recharge_diff.columns = ['id', "max_diff_recharge", "min_diff_recharge", "avg_diff_recharge", "sum_diff_recharge"]
    card_feature = pd.merge(card_feature, recharge_diff, how='left', on='id')
    del recharge_amount
    del recharge_diff
    
    # balance
    print('  -> balance feats')
    card = pd.DataFrame(card_records.groupby(['id'])['remainder'].max())
    card['remainder_min'] = card_records.groupby(['id'])['remainder'].min()
    card['remainder_avg'] = card_records.groupby(['id'])['remainder'].mean()
    card.columns = ['max_balance', 'min_balance', 'avg_balance']
    card = card.reset_index()
    card_feature = pd.merge(card_feature, card, how='left', on='id')
    del card
    
    print('  -> save')
    card_feature.to_csv(join(feature_path, 'card_features_'+ flag +'.csv'), index=False)

if __name__ == '__main__':
    extract_card_feature('train')
    extract_card_feature('test')
    extract_score_feature()
    extract_id_feature()
    transform_ranking_feature()
    
    