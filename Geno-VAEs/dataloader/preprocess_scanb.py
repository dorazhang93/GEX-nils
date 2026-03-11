import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from load_from_file import load_variantcall, load_gex,load_nkbc, load_identifiers, load_RNAseq_protocols
from sklearn.preprocessing import StandardScaler
from vis import scatter_plot_gene_matrix, plot_selected_genes
from pathlib import Path
import scipy.stats
import random
plt.rcParams.update({'font.size': 12})

rng = np.random.RandomState(0)

class Preprocesser:
    def __init__(self,
                 exclude_gene_percent: list = None,
                 train_val_split: float = 0.2,
                 split_policy: str = "random",
                 supervised: bool = False,
                 output_folder: str = "",
                 sanity_check = False,
                 aligned=False,
                 offset=0.1,):
        """
        @param exclude_gene_percent: percentage of genes with low std which will be excluded
        @param normalization: whether to normalize input, if True using quantileTransformer
        @param train_val_split: fraction of validation set
        """
        self.exclude_gene_percent= exclude_gene_percent
        self.split = train_val_split
        self.split_policy = split_policy
        self.supervised=supervised
        self.align = aligned
        self.output_folder=output_folder
        self.offset = offset
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        if sanity_check:
            print("sanity check ...")
            if self._sanity_check():
                print("Passed sanity check")
            else:
                raise ("Failed sanity check")
        self.nkbc, self.rba_to_protocol, self.cohort_ids = load_nkbc(), load_RNAseq_protocols(), load_identifiers() # return pd.DataFrame, dict, pd.DataFrame
        # (case_id,gene_id)
        self.gex, self.case_ids, self.gene_ids = load_gex(aligned=aligned) # return matrix, array(case_ids), array(gene_ids)

    def _output_folder(self):
        supervised_tag = "supervised" if self.supervised else "unsupervised"
        aligned_tag = "aligned" if self.align else "non-aligned"
        return f"_{aligned_tag}_{supervised_tag}_"
    def _sanity_check(self):
        pass
        return True

    # Log(FPKM+0.1)
    def log_transform(self):
        print(f" Log transform data of shape {self.gex.shape[0]} x {self.gex.shape[1]} with offset of {self.offset}")
        self.gex = np.log2(self.gex + self.offset)


    def _filter_low_variance_gene(self):
        if self.exclude_gene_percent is None:
            print("output shape", self.gex.shape)
            np.savetxt(self.output_folder + "/gene_ids.txt", self.gene_ids.reshape(-1, 1), fmt="%s", delimiter='\t')
            return

        gene_std = self.gex.std(axis=0)  # (case_id,gene_id)

        # top1.5k varying genes
        gene_idx_selected = np.argsort(gene_std)[::-1][:1500]
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder + "/top1.5k_varying_gene_index.npy", gene_idx_selected)

        # top15k varying genes
        gene_idx_selected = np.argsort(gene_std)[::-1][:15000]
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder+"/top15k_varying_gene_index.npy", gene_idx_selected)

        # top10k varying genes
        gene_idx_selected = np.argsort(gene_std)[::-1][:10000]
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder + "/top10k_varying_gene_index.npy", gene_idx_selected)

        # top2.5k varying genes
        gene_idx_selected = np.argsort(gene_std)[::-1][:2500]
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder + "/top2.5k_varying_gene_index.npy", gene_idx_selected)

        # ref Looket 2019 top 5000 genes and BH corrected p-value of critical level of 10%
        gene_idx_selected = np.argsort(gene_std)[::-1][:5000]
        dev_idx=((self.nkbc.diag_year!=2015) & (self.nkbc.diag_year!=2016) & (self.nkbc.diag_year!=2017) &
                 (~self.nkbc['LNM status'].isna()))
        gex_dev = self.gex[dev_idx]
        nkbc_dev=self.nkbc[dev_idx]
        print(f" development set has {dev_idx.sum()} cases")
        def pass_p_test_with_correction(selected_gene_idxs, gex, label):
            assert len(label) == len(gex)
            gex_pos = gex[label == 'Positive', :]
            gex_neg = gex[label == 'Negative', :]
            assert len(gex_pos) + len(gex_neg) == len(gex)
            ps = []
            for gene_idx in selected_gene_idxs:
                ps.append(scipy.stats.ttest_ind(gex_pos[:, gene_idx], gex_neg[:, gene_idx], equal_var=False).pvalue)
            ps = np.array(ps) * len(ps)
            rank = ps.argsort().argsort() + 1
            corrected_ps = ps / rank
            selected_gene_idxs = selected_gene_idxs[corrected_ps < 0.1]
            print(f"{len(selected_gene_idxs)} genes were selected with corrected p-value <0.1")

            fig, ax = plt.subplots(figsize=(6,3))
            descend_ord = np.argsort(corrected_ps)
            ax.plot(np.array(range(len(corrected_ps))), corrected_ps[descend_ord])
            ax.set_xlabel("Genes by ascending P values")
            ax.set_ylabel("Corrected P values")
            ax.set_title("Multiple t-test after Benjamin-Hochberg correction")
            ax.vlines(len(selected_gene_idxs), 0, 1.0, color='r',linestyles='--')
            ax.annotate(f"{len(selected_gene_idxs)} genes selected", xy=(len(selected_gene_idxs), 0.1),
                         xytext=(len(selected_gene_idxs)+500, 0.08),
                         arrowprops=dict(arrowstyle="->"))
            plt.savefig(self.output_folder+"/multiple-t-test_dist.png")

            return selected_gene_idxs

        gene_idx_selected = pass_p_test_with_correction(gene_idx_selected, gex_dev, nkbc_dev['LNM status'].values)
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder + "/looket2019_gene_index.npy", gene_idx_selected)

        # random selected 2500 genes
        random.seed(0)
        gene_idx_selected = np.array(random.sample(range(len(gene_std)), 2500))
        gene_idx_selected.sort()  # keep the original order of genes
        np.save(self.output_folder + "/random2.5k_gene_index.npy", gene_idx_selected)


        np.savetxt(self.output_folder+"/gene_ids.txt",self.gene_ids.reshape(-1,1), fmt="%s", delimiter='\t')

    def _normalize(self):
        """
        Transform gex data to normal distribution
        """
        print("sample wise normalization with StandardScale()")
        self.gex=(StandardScaler().fit_transform(self.gex.T)).T
        scatter_plot_gene_matrix(self.gex,self.output_folder+"/GEX_normalized.png")

    def _align_nkbc_samples(self):
        self.nkbc.index = self.nkbc.CaseName.values.tolist()
        rba_to_casename_map=dict(zip(self.cohort_ids.rba.values.tolist(), self.cohort_ids.CaseName.values.tolist()))
        casename_to_rba=dict(zip(self.cohort_ids.CaseName.values.tolist(), self.cohort_ids.rba.values.tolist()))
        aligned_nkbc=[]
        for case_rba in self.case_ids:
            aligned_nkbc.append(self.nkbc.loc[rba_to_casename_map[case_rba],:].values.tolist())

        self.nkbc = pd.DataFrame(aligned_nkbc,columns=self.nkbc.columns)

        # extract protocols
        protocols=[self.rba_to_protocol[casename_to_rba[pid]] for pid in self.nkbc.CaseName.values.tolist()]
        self.nkbc['libprotocol']=protocols
        # extract rba indexs
        rbas=[casename_to_rba[pid] for pid in self.nkbc.CaseName.values.tolist()]
        self.nkbc['rba']=rbas
    def _select_patient(self):
        if self.supervised:
            print("select cases for supervised pretraining")
            patient_selection= self.nkbc.prediciton_subset
            self.nkbc = self.nkbc[patient_selection]
            self.gex = self.gex[patient_selection]
            self.case_ids =self.case_ids[patient_selection]
        else:
            print("Use all cases for unsupervised pretraining")
    def _train_val_split(self):
        if self.split_policy == "random":
            print("Randomly split ...")
            gex_train, gex_val, case_train, case_val, nkbc_train, nkbc_val= train_test_split(self.gex, self.case_ids,
                                                                                               self.nkbc,
                                                                                               test_size=self.split,
                                                                                               random_state=rng)
        elif self.split_policy == "stratified":
            print("Stratified split based on SLNM status ... ")
            pop_list=self.nkbc['LNM status'].values # specify stratifing key
            gex_train, gex_val, case_train, case_val, nkbc_train, nkbc_val = train_test_split(self.gex, self.case_ids,
                                                                                               self.nkbc,
                                                                                                test_size=self.split,
                                                                                                random_state=rng,
                                                                                                stratify=pop_list)
        else:
            raise NotImplementedError(f"{self.split_policy} is not implemented" )
        return gex_train, gex_val, case_train, case_val, nkbc_train, nkbc_val

    def process(self):
        """
        preprocessing gex data
        """
        # 0. align nkbc data with the gex data in terms of the order of the rows/samples
        self._align_nkbc_samples()
        # 0. if supervised, select patients
        self._select_patient()
        # 1. log transform
        self.log_transform()
        # 2. filter out genes with low variance
        self._filter_low_variance_gene()
        import sys
        sys.exit()

        # 3. sample-wise normalization using StandardScale
        self._normalize()

        # 4. split val and train for unsupervised learning
        if self.supervised:
            np.save(self.output_folder + "/gex_all.npy", self.gex.astype(np.float32))
            np.savetxt(self.output_folder + "/case_all.txt", self.case_ids.reshape(-1, 1), fmt='%s', delimiter='\t')
            self.nkbc.to_csv(self.output_folder + "/nkbc_all.csv",index=False)
            plot_selected_genes()
            return

        gex_train, gex_val, case_train, case_val, nkbc_train, nkbc_val =self._train_val_split()

        # save data
        data_dict = {"gex": {"all": self.gex, "train": gex_train, "val": gex_val},
                     "case": {"all": self.case_ids, "train": case_train, "val": case_val},
                     "nkbc": {"all": self.nkbc, "train": nkbc_train, "val": nkbc_val}}

        for dtyp, v in data_dict.items():
            for split, value in v.items():
                if dtyp == "gex":
                    np.save(self.output_folder + f"/{dtyp}_{split}.npy", value.astype(np.float32))
                elif dtyp == "case":
                    np.savetxt(self.output_folder + f"/{dtyp}_{split}.txt", value.reshape(-1,1), fmt='%s', delimiter='\t')
                elif dtyp == "nkbc":
                    value.to_csv(self.output_folder + f"/{dtyp}_{split}.csv",index=False)
                else:
                    raise ValueError(f"Wrong data {dtyp}")



if __name__ == "__main__":
    params= {"exclude_gene_percent": [0, 15000, 10000, ], # if 0, only genes with zero variance will be excluded, if >1, select top n, if None no selection
             "output_folder":"/home/avesta/daqu/Projects/GEX/GEX_processed/gex/case9263_aligned_offset3",
             "aligned":True, # whether used data with the correction of rna-seq protocols
             "supervised":True, # whether it is self-supervised pretraining or supervised pretraining on clinical labels
             "split_policy":"random", # or "stratified if supervised pretraining
             "offset":0.1,
             }
    print(params)
    processer = Preprocesser(**params)
    processer.process()