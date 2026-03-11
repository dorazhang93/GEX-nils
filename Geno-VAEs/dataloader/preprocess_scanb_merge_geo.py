import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from load_from_file import load_gene_annotation, load_gex
from sklearn.preprocessing import StandardScaler
from vis import scatter_plot_gene_matrix
from pathlib import Path
import scipy.stats
import random
rng = np.random.RandomState(0)

gene_anno = load_gene_annotation()
geneID_to_HGNC = dict(zip(gene_anno["Gene.ID"].values, gene_anno["Gene.Name"].values))
HGNC_to_ensg = dict(zip(gene_anno["Gene.Name"].values, [x.split('.')[0] for x in gene_anno["Gene.ID"].values]))

class Preprocesser:
    def __init__(self,
                 train_val_split: float = 0.025,
                 output_folder: str = "",
                 offset=0.1,
                 gene_included: str = None):
        """
        @param exclude_gene_percent: percentage of genes with low std which will be excluded
        @param normalization: whether to normalize input, if True using quantileTransformer
        @param train_val_split: fraction of validation set
        """
        self.split = train_val_split
        self.output_folder=output_folder
        self.offset = offset
        self.gene_included = gene_included
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)


        # (case_id,gene_id)
        self.scanb_fpkm, self.scanb_case_ids, self.scanb_gene_ids = load_gex(aligned=True) # return matrix, array(case_ids), array(gene_ids)
        self.geo_fpkm_df = pd.read_csv('/home/avesta/daqu/Projects/GEX/archs4/fpkm_sample_qc_scanb_genes.csv',index_col=0) # sample x gene
        self.scanb_fpkm_df = None

        self.gex=None
        self.case_ids=None
        self.gene_ids=None

    def sum_duplicated_scanb_genes(self):
        scanb_genes = [x.split('.')[0] for x in self.scanb_gene_ids]
        scanb_fpkm_df = pd.DataFrame(self.scanb_fpkm, columns=scanb_genes)

        unique_cols = scanb_fpkm_df.columns.unique()
        data = []
        for col in unique_cols:
            seg = scanb_fpkm_df.loc[:, col].values
            if len(seg.shape) > 1:
                data.append(seg.sum(axis=1))
            elif len(seg.shape) == 1:
                data.append(seg)
            else:
                raise ValueError(f"Invalid gene id {col}")
        self.scanb_fpkm_df = pd.DataFrame(np.array(data).T, columns=unique_cols, index=self.scanb_case_ids)
        print("Summed up duplicated genes, resulted gex matrix", self.scanb_fpkm_df.values.shape)

    def save_scanb(self,merged_df):
        n_samples_scanb = len(self.scanb_case_ids)
        gex_scanb = merged_df.values[:n_samples_scanb, :]
        cases_scanb = merged_df.index[:n_samples_scanb]
        genes_scanb = merged_df.columns.values
        previous_saved_genes = np.loadtxt(self.output_folder + "/gene_ensgs.txt", dtype=str)
        assert np.array_equal(np.array(genes_scanb), np.squeeze(previous_saved_genes))
        gex_scanb = np.log2(gex_scanb + self.offset)
        gex_scanb = (StandardScaler().fit_transform(gex_scanb.T)).T
        np.save(self.output_folder + f"/scanb_gex.npy", gex_scanb.astype(np.float32))
        np.savetxt(self.output_folder + f"/scanb_cases9263.txt", np.array(cases_scanb).reshape(-1, 1), fmt='%s', delimiter='\t')

    def merge_scanb_geo(self):
        geo_genes = [HGNC_to_ensg[x] for x in self.geo_fpkm_df.columns]
        self.geo_fpkm_df.columns = geo_genes
        print("Shared common genes of size", len(set(geo_genes)&set(list(self.scanb_fpkm_df.columns))))
        assert len(set(list(self.geo_fpkm_df.index)) & set(list(self.scanb_fpkm_df.index)))==0
        print("Geo gex matrix", self.geo_fpkm_df.values.shape)

        merged_df = pd.concat([self.scanb_fpkm_df,self.geo_fpkm_df],ignore_index=False, sort=False)
        print("Merged scanb and Geo, resulted gex matrix", merged_df.values.shape)

        # self.save_scanb(merged_df)
        # import sys
        # sys.exit()

        self.gex = merged_df.values
        self.gene_ids = merged_df.columns.values
        self.case_ids = merged_df.index.values
        np.savetxt(self.output_folder + "/gene_ensgs.txt", self.gene_ids.reshape(-1, 1), fmt="%s", delimiter='\t')

    def select_genes(self):
        genes_in = np.loadtxt(self.gene_included, dtype=str)
        print(f"Loaded gene list of ", len(genes_in))
        genes_in_idx = np.isin(self.gene_ids,genes_in)
        self.gene_ids = self.gene_ids[genes_in_idx]
        self.gex =  (self.gex.T)[genes_in_idx].T
        print(f"Filtered in gex matrix of ", self.gex.shape)
        np.savetxt(self.output_folder + "/gene_ensgs.txt", self.gene_ids.reshape(-1, 1), fmt="%s", delimiter='\t')

        self.save_scanb(pd.DataFrame(self.gex, columns=self.gene_ids.tolist(), index=self.case_ids.tolist()))

    # Log(FPKM+0.1)
    def log_transform(self):
        print(f" Log transform data of shape {self.gex.shape[0]} x {self.gex.shape[1]} with offset of {self.offset}")
        self.gex = np.log2(self.gex + self.offset)

    def _normalize(self):
        """
        Transform gex data to normal distribution
        """
        print("sample wise normalization with StandardScale()")
        self.gex=(StandardScaler().fit_transform(self.gex.T)).T
        scatter_plot_gene_matrix(self.gex[:50000] if len(self.gex)>5e4 else self.gex,self.output_folder+"/GEX_normalized.png")

    def _train_val_split(self):
        print("Randomly split ...")
        gex_train, gex_val, case_train, case_val,= train_test_split(self.gex, self.case_ids,
                                                                                          test_size=self.split,
                                                                                          random_state=rng)

        return gex_train, gex_val, case_train, case_val

    def process(self):
        """
        preprocessing gex data
        """
        self.sum_duplicated_scanb_genes()

        self.merge_scanb_geo()

        if self.gene_included:
            self.select_genes()
        # 1. log transform
        self.log_transform()

        # 3. sample-wise normalization using StandardScale
        self._normalize()

        gex_train, gex_val, case_train, case_val =self._train_val_split()

        # save data
        data_dict = {"gex": {"all": self.gex, "train": gex_train, "val": gex_val},
                     "case": {"all": self.case_ids, "train": case_train, "val": case_val},}

        for dtyp, v in data_dict.items():
            for split, value in v.items():
                if dtyp == "gex":
                    np.save(self.output_folder + f"/{dtyp}_{split}.npy", value.astype(np.float32))
                elif dtyp == "case":
                    np.savetxt(self.output_folder + f"/{dtyp}_{split}.txt", value.reshape(-1,1), fmt='%s', delimiter='\t')
                else:
                    raise ValueError(f"Wrong data {dtyp}")



if __name__ == "__main__":
    params= {"output_folder":"/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo_over_nkbc",
             "offset":0.1,
             "gene_included":"/home/avesta/daqu/Projects/GEX/GEX_processed/gex/scanb_geo/t-test_cancerpathway_Gradcam_over_nkbc.txt"
             }
    print(params)
    processer = Preprocesser(**params)
    processer.process()