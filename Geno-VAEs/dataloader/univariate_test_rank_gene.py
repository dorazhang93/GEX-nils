import numpy as np
import scipy.stats

from load_from_file import load_nkbc,load_identifiers
import pandas as pd
import scipy
import pickle


def load_gex():
    gex=np.load("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/scanb_gex19657.npy")
    case_ids=np.loadtxt("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/scanb_cases9263.txt",dtype=str) # S000001.l.r.m.c.lib.g.k2.a.t
    gene_ids=np.loadtxt("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/gene_ensgs.txt", dtype=str)
    return gex, case_ids, gene_ids

# (case_id,gene_id)
gex, case_ids, gene_ids = load_gex()  # return matrix, array(case_ids), array(gene_ids)


def _align_nkbc_samples():
    nkbc, cohort_ids = load_nkbc(), load_identifiers()  # return pd.DataFrame, dict, pd.DataFrame
    nkbc.index = nkbc.CaseName.values.tolist()
    rba_to_casename_map = dict(zip(cohort_ids.rba.values.tolist(), cohort_ids.CaseName.values.tolist()))
    aligned_nkbc = []
    for case_rba in case_ids:
        aligned_nkbc.append(nkbc.loc[rba_to_casename_map[case_rba], :].values.tolist())

    nkbc = pd.DataFrame(aligned_nkbc, columns=nkbc.columns)
    nkbc['rba'] = case_ids
    return nkbc

nkbc = _align_nkbc_samples()
def univariate_test(gex_pos,gex_neg):
    assert gex_pos.shape[1]==gex_neg.shape[1]
    ttest_pvalues=[]
    wilcox_pvalues=[]
    for i in range(gex_pos.shape[1]):
        ttest_pvalues.append(scipy.stats.ttest_ind(gex_pos[:,i],gex_neg[:,i],equal_var=False).pvalue)
        wilcox_pvalues.append(scipy.stats.ranksums(gex_pos[:,i],gex_neg[:,i]).pvalue)
    print(f"Pearson correlation between ttest and wilcox is {scipy.stats.pearsonr(ttest_pvalues,wilcox_pvalues)[0]}")

    return wilcox_pvalues

def univariate_test_Tsize(gex_matrix, labels):
    assert gex_matrix.shape[0]==len(labels)
    pearsonr_values=[]
    for i in range(gex_matrix.shape[1]):
        pearsonr_values.append(scipy.stats.pearsonr(gex_matrix[:,i],labels)[0])
    return pearsonr_values

def load_subtypes():
    cohort_identifier_file = '/home/avesta/daqu/Projects/GEX/LN_cohort_v3.0_data/LN_cohort_v3.0/LNcohort.txt'
    cohort_identifier = pd.read_csv(cohort_identifier_file, sep='\t')
    CaseName2rba = dict(zip(cohort_identifier.CaseName.values, cohort_identifier.rba.values))

    nkbc_data = pd.read_csv("/home/avesta/daqu/Projects/GEX/GEX_processed/nkbc/nkbc_recoded_9263.csv")
    data = nkbc_data[['Clinical subtype', 'CaseName']]
    data['CaseName'] = [CaseName2rba[cid] for cid in data.CaseName.values]
    data = data.rename(columns={"CaseName": 'case_id',
                                'Clinical subtype': 'subtype'})
    return data
pvalues_task={}
case_subtype_df=load_subtypes()
case2subtype =dict(zip(case_subtype_df.case_id.values,case_subtype_df.subtype.values))

for task in ['LNM status','SLNM status','Tumor size', '5-year DRF status']:
    pvalues_task[task]={}
    dev_idx=((nkbc.diag_year!=2015) & (nkbc.diag_year!=2016) & (nkbc.diag_year!=2017) &
                 (~nkbc[task].isna()) & (nkbc.prediciton_subset))
    print(f"{task} development set has {dev_idx.sum()} cases")

    gex_matrix = gex[dev_idx]
    labels=nkbc.loc[dev_idx,task].values
    assert gex_matrix.shape[0] == len(labels)

    # continues variable
    if task =='Tumor size':
        pvalue_genes=univariate_test_Tsize(gex_matrix,labels)
    else:
        if task=='5-year DRF status':
            gex_pos = gex_matrix[labels == 'Relapse', :]
            gex_neg = gex_matrix[labels == 'No relapse', :]
        else:
            gex_pos = gex_matrix[labels == 'Positive', :]
            gex_neg = gex_matrix[labels == 'Negative', :]
        pvalue_genes=univariate_test(gex_pos,gex_neg)
    pvalues_task[task]['All']=pvalue_genes

    for subtype in ['ER+HER2-','HER2+','TNBC']:
        caseInSubtype=np.array([case2subtype[x]==subtype for x in case_ids])
        assert len(caseInSubtype)==len(dev_idx)
        subgroup_idx= (caseInSubtype & dev_idx)

        gex_matrix = gex[subgroup_idx]
        labels = nkbc.loc[subgroup_idx, task].values
        assert gex_matrix.shape[0] == len(labels)
        print(f"{subtype} has {subgroup_idx.sum()} cases")

        # continues variable
        if task == 'Tumor size':
            pvalue_genes = univariate_test_Tsize(gex_matrix, labels)
        else:
            if task == '5-year DRF status':
                gex_pos = gex_matrix[labels == 'Relapse', :]
                gex_neg = gex_matrix[labels == 'No relapse', :]
            else:
                gex_pos = gex_matrix[labels == 'Positive', :]
                gex_neg = gex_matrix[labels == 'Negative', :]
            pvalue_genes = univariate_test(gex_pos, gex_neg)
        pvalues_task[task][subtype] = pvalue_genes

with open("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/scanb_wilcoxPvalues.pickle","wb") as f:
    pickle.dump(pvalues_task,f)

# rank genes based on SLNM and LNM importance
ranks= {}
for subtype in ['ER+HER2-', 'HER2+','TNBC','All']:
    rank_subty=[]
    for task in ["SLNM status", "LNM status"]:
        rank_subty.append(np.argsort(np.argsort(pvalues_task[task][subtype])))
    rank_subty=np.argsort(np.argsort(np.array(rank_subty).mean(axis=0)))
    ranks[subtype]=rank_subty


rank_pvalue_tasks={'rank_NM':{'full_rank':ranks['All']}}
rank_pvalue_tasks['rank_NM_ER+HER2-']={'full_rank':ranks['ER+HER2-']}
rank_pvalue_tasks['rank_NM_HER2+']={'full_rank':ranks['HER2+']}
rank_pvalue_tasks['rank_NM_TNBC']={'full_rank':ranks['TNBC']}

rank_pvalue_tasks['rank_Tsize']={'full_rank':np.argsort(np.argsort(pvalues_task['Tumor size']['All']))}
rank_pvalue_tasks['rank_DRF']={'full_rank':np.argsort(np.argsort(pvalues_task['5-year DRF status']['All'])),}

for task in ['NM','NM_ER+HER2-','NM_HER2+','NM_TNBC','Tsize','DRF']:
    rank=rank_pvalue_tasks[f"rank_{task}"]['full_rank']
    ascent_idx=np.argsort(rank)
    for topk in 10*(2**np.array(range(11))):
        top_genes = gene_ids[ascent_idx[:topk]]
        rank_pvalue_tasks[f"rank_{task}"][f'top_{topk}']=top_genes

with open("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/scanb_top_univariate_differ_genelist.pickle","wb") as f:
    pickle.dump(rank_pvalue_tasks,f)






