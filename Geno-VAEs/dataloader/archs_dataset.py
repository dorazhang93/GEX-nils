import archs4py as a4
import numpy as np
import pandas as pd
from load_from_file import load_gex
import matplotlib.pyplot as plt
import random
random.seed(0)

# construct Gene_symbol to HGNC map
scanb_gene_annotation = pd.read_csv(
    '/home/avesta/daqu/Projects/GEX/LN_cohort_v3.0_data/LN_cohort_v3.0_GEX/Gene.ID.ann.mergeV27.txt', sep='\t')
geneID_to_HGNC = dict(zip(scanb_gene_annotation["Gene.ID"].values, scanb_gene_annotation["Gene.Name"].values))
HGNC_to_ensg = dict(zip(scanb_gene_annotation["Gene.Name"].values, [x.split('.')[0] for x in scanb_gene_annotation["Gene.ID"].values]))
file="/home/avesta/daqu/Projects/GEX/archs4/geo_human_v1.h5"
a4.ls(file)
"""
data                      
│ expression            uint32 | (39376, 340713)
meta                      
│ genes                     
│   gene_symbol           str    | (39376,)
│ samples                   
│   channel_count         str    | (340713,)
│   characteristics_ch1   str    | (340713,)
│   contact_address       str    | (340713,)
│   contact_city          str    | (340713,)
│   contact_country       str    | (340713,)
│   contact_institute     str    | (340713,)
│   contact_name          str    | (340713,)
│   contact_zip           str    | (340713,)
│   data_processing       str    | (340713,) "The quality of the raw reads were assessed..."
│   extract_protocol_ch1  str    | (340713,)
│   geo_accession         str    | (340713,)
│   instrument_model      str    | (340713,)
│   last_update_date      str    | (340713,)
│   library_selection     str    | (340713,)
│   library_source        str    | (340713,)
│   library_strategy      str    | (340713,)
│   molecule_ch1          str    | (340713,)
│   organism_ch1          str    | (340713,)
│   platform_id           str    | (340713,)
│   relation              str    | (340713,)
│   sample                str    | (340713,)
│   series_id             str    | (340713,)
│   source_name_ch1       str    | (340713,)
│   status                str    | (340713,)
│   submission_date       str    | (340713,)
│   taxid_ch1             str    | (340713,)
│   title                 str    | (340713,)
│   type                  str    | (340713,)
"""

""" @@@@ Sample selection

"""
def check_missing_genes(gene_geo,gene_scanb,geneID_to_HGNC):
    looket2019_genes=gene_scanb[np.load("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/looket2019_gene_index.npy")]
    cancer_pathways=np.load("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/cancer_pathways.npy")
    cancer_genes = gene_scanb[cancer_pathways.sum(axis=0)>0]
    gradcam_important_genes=np.loadtxt("/home/avesta/daqu/Projects/GEX/code/results/GradCAM/allgene-nkbc_transformer/genes_outperform_at_least_one_nkbc_Alltasks.txt",dtype=str)
    gradcam_important_genes=[geneID_to_HGNC[x] for x in gradcam_important_genes]
    top10k_varying_genes=gene_scanb[np.load("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/top10k_varying_gene_index.npy")]

    print(f"{len(set(looket2019_genes.tolist())-set(gene_geo))/len(looket2019_genes)} |"
          f"{len(set(looket2019_genes.tolist())-set(gene_geo))} out of {len(looket2019_genes)} looket2019 genes not found in GEO")
    print(f"{len(set(cancer_genes.tolist())-set(gene_geo))/len(cancer_genes)} |"
          f"{len(set(cancer_genes.tolist())-set(gene_geo))} out of {len(cancer_genes)} cancer pathway genes not found in GEO")
    print(f"{len(set(gradcam_important_genes)-set(gene_geo))/len(gradcam_important_genes)} |"
          f"{len(set(gradcam_important_genes)-set(gene_geo))} out of {len(gradcam_important_genes)} gradcam important genes not found in GEO")
    print(f"{len(set(top10k_varying_genes.tolist())-set(gene_geo))/len(top10k_varying_genes)} |"
          f"{len(set(top10k_varying_genes.tolist())-set(gene_geo))} out of {len(top10k_varying_genes)} top10k varying genes not found in GEO")
    print(f"{len(set(gene_scanb.tolist())-set(gene_geo))/len(gene_scanb)} |"
          f"{len(set(gene_scanb.tolist())-set(gene_geo))} out of {len(gene_scanb)} scanb genes not found in GEO")


def extract_scanb_gene_list():

    # SCANB gene HGNC ids
    gene_scanb = np.loadtxt("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/gene_ids.txt", dtype=str)
    gene_scanb = np.array([geneID_to_HGNC[x] for x in gene_scanb])

    # GEO gene HGNC ids
    gene_geo = a4.meta.get_meta_gene_field(file,'gene_symbol')

    check_missing_genes(gene_geo,gene_scanb,geneID_to_HGNC)

    genes_include = list(set(gene_scanb.tolist()) & set(gene_geo))
    genes_idxs = np.argwhere(np.isin(np.array(gene_geo),genes_include)).ravel()
    genes_include =np.array(gene_geo)[genes_idxs]
    assert len(genes_idxs)==len(genes_include)
    print(f'{len(genes_idxs)} genes were included from GEO dataset')
    return genes_idxs, genes_include


def sample_QC(genes_idxs):
    # select samples with > 75% none-zero genes to exclude misclassified or contaminated RNA-seq data
    samples = a4.meta.get_meta_sample_field(file, 'sample')
    df = a4.data.index(file,list(range(len(samples))), genes_idxs) # pd.DataFrame indexed by genes and columned by sample
    gex_raw_count = df.values.T #sample x gene
    print("GEO raw count matrix", gex_raw_count.shape)
    assert gex_raw_count.min()>=0
    non_zero_gene_count = np.count_nonzero(gex_raw_count, axis=1)
    qc_sample_index = non_zero_gene_count> int(0.75*len(genes_idxs))
    print(f"{qc_sample_index.sum()} samples passed QC")
    return df.loc[:,qc_sample_index]


def raw_count_to_FPKM(raw_df):
    gene_ids = raw_df.index
    samples = raw_df.columns
    gene_engs=[HGNC_to_ensg[x] for x in gene_ids]
    gene_length_df = pd.read_csv('/home/avesta/daqu/Projects/GEX/BulkFormer/data/gene_length_df.csv')
    gene_length_dict = gene_length_df.set_index('ensg_id')['length'].to_dict()
    gene_length_kb = np.array([gene_length_dict.get(gene, 2481) for gene in gene_engs]) # set default length as the mean value

    raw_counts=raw_df.values.T # samples x genes
    rate=raw_counts/gene_length_kb
    sum_per_sample = raw_counts.sum(axis=1)
    fpkm = rate / sum_per_sample[:,None] *1e9
    fpkm_df=pd.DataFrame(fpkm,index=samples,columns=gene_ids)
    return fpkm_df

def log_transform(matrix):
    return np.log2(matrix+0.1)

def compare_scatter_plot_gene_matrix(data_s_df,data_g_df,output_file):
    # data shape (case, gene)
    geo_matrix=data_g_df.values

    # geo gene descending order
    geo_gene_descent=np.argsort(geo_matrix.mean(axis=0))[::-1]
    geo_case_descent=np.argsort(geo_matrix.mean(axis=1))[::-1]

    assert len(data_s_df.columns) > len(data_g_df.columns)
    gene_geo = data_g_df.columns
    # sort scanb genes by first geo genes and then extra
    scanb_in_df = data_s_df.loc[:,gene_geo]
    print(f"scanb_in_df shape", scanb_in_df.values.shape)
    scanb_out_df = data_s_df.loc[:,~data_s_df.columns.isin(gene_geo)]
    print(f"scanb_out_df shape", scanb_out_df.values.shape)
    data_s_df = pd.concat([scanb_in_df,scanb_out_df],axis=1)
    scanb_matrix = data_s_df.values
    print(f"scanb_matrix shape", scanb_matrix.shape)
    scanb_matrix_in = scanb_in_df.values
    scanb_matrix_out = scanb_out_df.values
    scanb_gene_out_descent = np.argsort(scanb_matrix_out.mean(axis=0))[::-1]
    scanb_case_descent=np.argsort(scanb_matrix.mean(axis=1))[::-1]

    """ scatter plots"""
    fig, axs=plt.subplots(2,2,figsize=(12,12))
    # plot geo expression
    g_x=np.repeat(np.array(range(len(geo_gene_descent))),len(geo_case_descent))
    axs[0,0].scatter(x=g_x,y=geo_matrix.T[geo_gene_descent].reshape(-1), s=0.1,linewidth=0)
    axs[0,0].set(xlabel='gene id (sorted by mean)',ylabel='GEO expression level')
    c_x=np.repeat(np.array(range(len(geo_case_descent))),len(geo_gene_descent))
    axs[0,1].scatter(x=c_x,y=geo_matrix[geo_case_descent].reshape(-1), s=0.1,linewidth=0)
    axs[0,1].set(xlabel='case id (sorted by mean)',ylabel='GEO expression level')

    # plot scanb with first included genes sorted by GEO mean and then extra genes
    g_x_in = np.repeat(np.array(range(len(geo_gene_descent))), scanb_matrix_in.shape[0])
    axs[1, 0].scatter(x=g_x_in, y=scanb_matrix_in.T[geo_gene_descent].reshape(-1), s=0.1, linewidth=0)
    g_x_out = np.repeat(np.array(range(len(geo_gene_descent),scanb_matrix.shape[1])), scanb_matrix_out.shape[0])
    print(geo_gene_descent.shape, scanb_matrix.shape,g_x_out.shape, scanb_matrix_out.shape,scanb_gene_out_descent.shape)
    axs[1, 0].scatter(x=g_x_out, y=scanb_matrix_out.T[scanb_gene_out_descent].reshape(-1), s=0.1, linewidth=0)
    axs[1, 0].set(xlabel='gene id (sorted by mean)', ylabel='SCANB expression level')

    c_x = np.repeat(np.array(range(len(scanb_case_descent))), scanb_matrix.shape[1])
    axs[1, 1].scatter(x=c_x, y=scanb_matrix[scanb_case_descent].reshape(-1), s=0.1, linewidth=0)
    axs[1, 1].set(xlabel='case id (sorted by mean)', ylabel='SCANB expression level')
    plt.savefig(output_file)

def geo_vs_scanb_gex(scanb_fpkm_df,geo_fpkm_df):
    output_folder="/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo"
    # compare the FPKM between geo and scanb datasets
    compare_scatter_plot_gene_matrix(scanb_fpkm_df,geo_fpkm_df,f"{output_folder}/compare_fpkm.png")

    # compare the sample-wise normalized log transformation
    scanb_fpkm_df= pd.DataFrame(log_transform(scanb_fpkm_df.values),columns=scanb_fpkm_df.columns,index=scanb_fpkm_df.index)
    geo_fpkm_df= pd.DataFrame(log_transform(geo_fpkm_df.values),columns=geo_fpkm_df.columns, index=geo_fpkm_df.index)
    compare_scatter_plot_gene_matrix(scanb_fpkm_df,geo_fpkm_df,f"{output_folder}/compare_logfpkm.png")


def merge_duplicated_scanb_genes(df):
    samples=df.index
    unique_cols = df.columns.unique()
    data=[]
    for col in unique_cols:
        seg = df.loc[:,col].values
        if len(seg.shape) >1:
            data.append(seg.sum(axis=1))
        elif len(seg.shape) ==1:
            data.append(seg)
        else:
            raise ValueError(f"Invalid gene id {col}")
    return pd.DataFrame(np.array(data).T, columns=unique_cols, index=samples)




genes_idxs, genes_include=extract_scanb_gene_list()

geo_raw_count_df = sample_QC(genes_idxs) # genes x samples

geo_fpkm_df = raw_count_to_FPKM(geo_raw_count_df)
geo_fpkm_df.to_csv('/home/avesta/daqu/Projects/GEX/archs4/fpkm_sample_qc_scanb_genes.csv')

# geo_fpkm_df = pd.read_csv('/home/avesta/daqu/Projects/GEX/archs4/fpkm_sample_qc_scanb_genes.csv',index_col=0) # sample x gene
print(geo_fpkm_df.columns,geo_fpkm_df.index)
random_20k_sample_idx = random.sample(list(geo_fpkm_df.index),20000)
print(len(random_20k_sample_idx))
geo_fpkm_df = geo_fpkm_df.loc[random_20k_sample_idx,:]
gene_engs=[HGNC_to_ensg[x] for x in geo_fpkm_df.columns]
geo_fpkm_df.columns = gene_engs
print(geo_fpkm_df.columns,geo_fpkm_df.index)

scanb_fpkm, _, scanb_genes= load_gex(aligned=True) # sample x gene
scanb_genes = [x.split('.')[0] for x in scanb_genes]
scanb_fpkm_df = pd.DataFrame(scanb_fpkm,columns=scanb_genes)
scanb_fpkm_df = merge_duplicated_scanb_genes(scanb_fpkm_df)
print(scanb_fpkm_df.columns, scanb_fpkm_df.index)
geo_vs_scanb_gex(scanb_fpkm_df,geo_fpkm_df)









