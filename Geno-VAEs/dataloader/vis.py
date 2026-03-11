import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from load_from_file import load_ind_varcall, load_identifiers, load_gene_annotation, load_gex
import pandas as pd

def scatter_plot_gene_matrix(data,output_file):
    # data shape (case, gene)
    gene_std_descent=np.argsort(data.std(axis=0))[::-1]
    case_std_descent=np.argsort(data.std(axis=1))[::-1]

    """ scatter plots"""
    fig, axs=plt.subplots(1,2,figsize=(12,6))
    g_x=np.repeat(np.array(range(len(gene_std_descent))),len(case_std_descent))
    axs[0].scatter(x=g_x,y=data.T[gene_std_descent].reshape(-1), s=0.1,linewidth=0)
    axs[0].set(xlabel='gene id (sorted by std)',ylabel='gene expression level')
    c_x=np.repeat(np.array(range(len(case_std_descent))),len(gene_std_descent))
    axs[1].scatter(x=c_x,y=data[case_std_descent].reshape(-1), s=0.1,linewidth=0)
    axs[1].set(xlabel='case id (sorted by std)',ylabel='gene expression level')
    plt.savefig(output_file)


def plot_selected_genes():
    gex=np.load("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/gex_all.npy")
    # plot top15k
    gene_select_index = np.load(
        "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/top15k_varying_gene_index.npy")
    scatter_plot_gene_matrix(gex.T[gene_select_index].T,"/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/GEX_normalized_top15k.png")
    # plot top10k
    gene_select_index = np.load(
        "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/top10k_varying_gene_index.npy")
    scatter_plot_gene_matrix(gex.T[gene_select_index].T,
                             "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/GEX_normalized_top10k.png")
    # plot random 2.5k
    gene_select_index = np.load(
        "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/random2.5k_gene_index.npy")
    scatter_plot_gene_matrix(gex.T[gene_select_index].T,
                             "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/GEX_normalized_random2.5k.png")
    # plot looket2019
    gene_select_index = np.load(
        "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/looket2019_gene_index.npy")
    scatter_plot_gene_matrix(gex.T[gene_select_index].T,
                             "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/GEX_normalized_looket2019.png")

def plot_gex_mutation(geneCount_dict, gene_id2name):
    # load identifiers
    cohort_identifiers=load_identifiers()
    rba_to_SID=dict(zip(cohort_identifiers.rba.values, cohort_identifiers.SpecimenName.values))
    # patient filters included in the study cohort
    cases_included_rba = np.loadtxt(
        "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/case_all.txt", dtype=str)
    cases_included_SID=[rba_to_SID[x] for x in cases_included_rba]
    # load mutation data
    mutation_df=load_ind_varcall()
    mutation_df=mutation_df[mutation_df.serialID.isin(cases_included_SID)].reset_index(drop=True)
    mutation_count_gene=mutation_df.groupby(['geneID']).agg(Count=('geneID','count'))
    # load gex data
    gex = np.load("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/gex_all.npy") #(n_samples, n_genes)
    gene_ids = np.loadtxt("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/gene_ids.txt",
                          dtype=str)

    mutation_gwise_count=np.array([mutation_count_gene.loc[g, 'Count'] if g in mutation_count_gene.index else 0 for g in gene_ids])/1000
    for pathway_set, pathway_genecount in geneCount_dict.items():
        pathway_gene_idxs = pathway_genecount>0
        gex_in_pathway=(gex.T[pathway_gene_idxs]).T
        gex_out_pathway=(gex.T[~pathway_gene_idxs]).T
        gex_in_std = gex_in_pathway.std(axis=0)
        gex_out_std = gex_out_pathway.std(axis=0)
        mutation_in_pathway=mutation_gwise_count[pathway_gene_idxs]
        mutation_out_pathway=mutation_gwise_count[~pathway_gene_idxs]
        genecount_in_pathway=pathway_genecount[pathway_gene_idxs]
        sort_in_gene=np.argsort(genecount_in_pathway)[::-1] # genes in pathways were sorted by frequency of present in pathways
        sort_out_gene=np.argsort(gex_out_std)[::-1] # genes not in pathways were sorted by gex variations among samples
        max_in_gene_arg = np.argsort(mutation_in_pathway)[::-1][0]
        snd_in_gene_arg = np.argsort(mutation_in_pathway)[::-1][1]
        max_out_gene_arg = np.argsort(mutation_out_pathway)[::-1][0]
        snd_out_gene_arg = np.argsort(mutation_out_pathway)[::-1][1]
        max_in_gene = gene_id2name[gene_ids[pathway_gene_idxs][max_in_gene_arg]]
        snd_in_gene = gene_id2name[gene_ids[pathway_gene_idxs][snd_in_gene_arg]]
        max_out_gene = gene_id2name[gene_ids[~pathway_gene_idxs][max_out_gene_arg]]
        snd_out_gene = gene_id2name[gene_ids[~pathway_gene_idxs][snd_out_gene_arg]]

        x_in=np.array(range(len(genecount_in_pathway)))
        x_out=np.array(range(len(genecount_in_pathway),len(pathway_genecount)))
        fig, axs = plt.subplots(figsize=(8, 6))
        axs.plot(x_in, gex_in_std[sort_in_gene], label='In-pathway genes | gex variation')
        axs.plot(x_in, -mutation_in_pathway[sort_in_gene], label='In-pathway genes | mutation count')
        axs.plot(x_out,gex_out_std[sort_out_gene], label='Out-pathway genes | gex variation')
        axs.plot(x_out, -mutation_out_pathway[sort_out_gene], label='Out-pathway genes | mutation count')

        # name the most frequent gene name
        axs.annotate(max_in_gene, xy=(np.where(mutation_in_pathway[sort_in_gene]==mutation_in_pathway[max_in_gene_arg])[0][0], -3),
                     xytext=(np.where(mutation_in_pathway[sort_in_gene]==mutation_in_pathway[max_in_gene_arg])[0][0]+1000, -3),
                     arrowprops=dict(arrowstyle="->"))
        axs.annotate(snd_in_gene, xy=(np.where(mutation_in_pathway[sort_in_gene]==mutation_in_pathway[snd_in_gene_arg])[0][0], -2),
                     xytext=(np.where(mutation_in_pathway[sort_in_gene]==mutation_in_pathway[snd_in_gene_arg])[0][0]+1000, -2),
                     arrowprops=dict(arrowstyle="->"))
        axs.annotate(max_out_gene,
                     xy=(np.where(mutation_out_pathway[sort_out_gene] == mutation_out_pathway[max_out_gene_arg])[0][0]+len(genecount_in_pathway), -1),
                     xytext=(
                     np.where(mutation_out_pathway[sort_out_gene] == mutation_out_pathway[max_out_gene_arg])[0][0]+len(genecount_in_pathway) + 1000, -1),
                     arrowprops=dict(arrowstyle="->"))
        axs.annotate(snd_out_gene,
                     xy=(np.where(mutation_out_pathway[sort_out_gene] == mutation_out_pathway[snd_out_gene_arg])[0][0]+len(genecount_in_pathway), -0.5),
                     xytext=(np.where(mutation_out_pathway[sort_out_gene] == mutation_out_pathway[snd_out_gene_arg])[0][0]+len(genecount_in_pathway)+1000, -0.5),
                     arrowprops=dict(arrowstyle="->"))

        axs.legend()
        plt.ylabel('mutation counts (/500) | gex variation')
        plt.xlabel('gene id (sorted by frequency of pathway presentation, gex variation)')
        plt.savefig(
            f"/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/{pathway_set}_gex_mutation.png")




def plot_pathway_connection():
    gene_ids = np.loadtxt("/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case6836_aligned/gene_ids.txt",
                          dtype=str)
    gene_anno = load_gene_annotation()
    gene_id2name = dict(zip(gene_anno['Gene.ID'].values, gene_anno['Gene.Name']))
    # shape (num_pathway, num_genes)
    # load 39 cancer pathways
    cancer_pathways=np.load("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/cancer_pathways.npy")
    genCount_cancerPathway= cancer_pathways.sum(axis=0)
    # load 2541 reactome pathways
    reactome_pathways=np.load("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/reactome_pathways.npy")
    genCount_reactomePathway = reactome_pathways.sum(axis=0)
    # load 1426 pathformer pathways
    pathformer_pathways=np.load("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/pathformer_pathways.npy")
    genCount_pathformerPathway = pathformer_pathways.sum(axis=0)
    # load 3476 pathways all above
    all_pathways=np.load("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/all_pathways.npy")
    genCount_allPathway = all_pathways.sum(axis=0)

    # plot pathway mutation and gex expression levels
    plot_gex_mutation({'cancer_pathways':genCount_cancerPathway,
                       'reactome_pathways':genCount_reactomePathway,
                       'pathformer_pathways':genCount_pathformerPathway,
                       'all_pathways':genCount_allPathway,
                       },
                      gene_id2name
                      )

    # plot distribution of gene Counts
    geneCount_descent=np.argsort(genCount_allPathway)[::-1]
    geneCount_descent_cancer=np.argsort(genCount_cancerPathway)[::-1]
    zero_count_idx=None
    for i,idx in enumerate(geneCount_descent):
        if genCount_allPathway[idx]==0:
            zero_count_idx=i
            break
    gene_x = np.array(range(len(genCount_allPathway)))
    fig, axs = plt.subplots(figsize=(8, 6))
    axs.plot(gene_x,genCount_allPathway[geneCount_descent], label=f'all pathways (n=3,476 pathways, '
                                                                  f'{(genCount_allPathway>0).sum()} genes, {genCount_allPathway.sum()} connections)',alpha=0.5)
    axs.plot(gene_x,genCount_pathformerPathway[geneCount_descent], label=f'pathformer pathways (n=1,426 pathways, '
                                                                         f'{(genCount_pathformerPathway>0).sum()} genes, {genCount_pathformerPathway.sum()} connections)',alpha=0.5)
    axs.plot(gene_x, genCount_reactomePathway[geneCount_descent], label=f'reactome pathways (n=2,541 pathways, {(genCount_reactomePathway>0).sum()} genes, {genCount_reactomePathway.sum()} connections)',
             alpha=0.5)
    axs.plot(gene_x, genCount_cancerPathway[geneCount_descent], label=f'cancer pathways (n=39 pathways, {(genCount_cancerPathway>0).sum()} genes, {genCount_cancerPathway.sum()} connections)',
             alpha=0.5)
    axs.vlines(zero_count_idx,0,500, ls='--')
    # name the most frequent gene name
    axs.annotate(",".join([gene_id2name[gene_ids[geneCount_descent[i]]] for i in range(10)]), xy=(5, 200), xytext=(2500, 75),
                 arrowprops=dict(arrowstyle="->"))
    axs.annotate(",".join([gene_id2name[gene_ids[geneCount_descent_cancer[i]]] for i in range(10)]), xy=(5, 10),
                 xytext=(2500, 40),
                 arrowprops=dict(arrowstyle="->"))
    axs.legend()
    plt.yscale('symlog')
    plt.ylabel('Gene counts in pathway dataset')
    plt.xlabel('Gene id (sorted by gene count in all pathways)')
    plt.savefig("/home/avesta/daqu/Projects/GEX/code/ClinTab-DL/bin/gene_pathways/geneCounts_overlap_between_pathway_dataset.png")


def plot_gex_mean_var():
    gex, _,_ = load_gex(aligned=True)
    offsets = [10]
    output_folder = "/home/avesta/daqu/Projects/GEX/GEX_processed/gex/vis"

    for v in offsets:
        gex_log = np.log2(gex+v)
        gex_log_mean = gex_log.mean(axis=0)
        gex_log_var = gex_log.std(axis=0)

        idx_5k_var = np.argsort(gex_log_var)[::-1][5000]


        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(gex_log_mean, gex_log_var, s=1)
        plt.hlines(gex_log_var[idx_5k_var], min(gex_log_mean), max(gex_log_mean), ls='--')
        plt.xlabel("mean")
        plt.ylabel("std")
        plt.title(f"GEX data with log2 offset = {v}")
        plt.savefig(f"{output_folder}/gex_log2offset_{v}.png")



# plot_gex_mean_var()

