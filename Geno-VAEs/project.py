from config import *
from models import *
from dataloader import VAEDataset
import torch
from tqdm import tqdm
from glob import glob
import numpy as np

MODEL_FILE_ = "last*"
config, out_dir, _ = load_config()
# For reproducibility
torch.manual_seed(config['exp_params']['manual_seed'])

gpus = config['trainer_params']['gpus']
device = torch.device("cuda:%d" % gpus[0] if len(gpus) != 0 else "cpu")

def mse_mask(p,q,m):
    assert p.shape == q.shape
    mse = (p-q)**2
    assert mse.shape == m.shape
    return np.sum(mse*m)/np.sum(m),np.sum(m)/(m.size)


def CE_mask(p,q,m):
    assert p.shape == q.shape
    ce = -p*np.log(q)
    assert ce.shape == m.shape
    return np.sum(ce*m)/np.sum(m),np.sum(m)/(m.size)

def load_model():
    """
    load pretrained model from last.ckpt. Remove "model" in front of each parameter name.
    @return:
    @rtype:
    """
    print(config)
    checkpoint_path = glob(str(out_dir) + f"/checkpoints/{MODEL_FILE_}")[0]
    meta_load = torch.load(checkpoint_path)["state_dict"]
    meta_new = {}
    for k, v in meta_load.items():
        if k.startswith("model"):
            meta_new[k[6:]] = v
    model = vae_models[config['model_params']['name']](**config['model_params']).to(device)
    model.load_state_dict(meta_new)
    return model


def inference_all(save_recons=False):
    model = load_model()
    model.eval()

    # load data
    dataset = VAEDataset(**config["data_params"], pin_memory=len(gpus) != 0,
                         aux_on=config['model_params']['aux'])
    dataset.setup("predict")
    data_loader, meta = dataset.predict_dataloader()
    print(meta.columns)

    # project all split
    reconstructs = []
    latents = []
    losses=[]
    with torch.no_grad():
        for geno in tqdm(data_loader):
            if config['model_params']['aux'] == True:
                geno = [x.to(device) for x in geno]
            else:
                geno = geno.to(device)

            if config['model_params']['name'] == "iVAE":
                recons, input, z, post_dist, prior_dist = model(geno)
                loss = model.loss_function(recons, input, z, post_dist, prior_dist)
            elif config['model_params']['name'] == "idVAE":
                recons, input, z, recon_u, aux, post_dist, prior_dist = model(geno)
                loss = model.loss_function(recons, input, z, recon_u, aux, post_dist, prior_dist)
            elif config['model_params']['name'] == 'AutoEncoder':
                recons, input, z, = model(geno)
                loss = model.loss_function(recons, input, z)
            else:
                if config['model_params']['aux'] == True:
                    recons, input, z, latent_dist, aux, aux_lg = model(geno)
                    loss = model.loss_function(recons, input, z, latent_dist, aux, aux_lg)
                else:
                    recons, input, z, latent_dist = model(geno)
                    loss = model.loss_function(recons, input, z, latent_dist)
            if save_recons:
                reconstructs.append(recons.detach().cpu())
            latents.append(z.detach().cpu())
            losses.append(loss["loss"].detach().cpu())
        print(f"Average loss :{np.array(losses).mean()}")
        latents = np.concatenate(latents, axis=0)
        np.save(out_dir / "all_latent_features.npy", latents)
        np.savetxt(out_dir / "case_ids.txt", meta.CaseName.values, fmt="%s")
        if save_recons:
            reconstructs = np.concatenate(reconstructs, axis=0)
            np.save(out_dir / "all_reconstruction.npy", reconstructs)


def inference_val_missing(sparsifies=None):
    """
    Reconstruct genotypes of validation set, with model.train() setting. Missing value -1 generated
    by sparsify will be investigated. Two methods of missing imputation thus could be compared
    @param sparsify:
    @type sparsify:
    @return geno_true, geno_recons,mf_allele,ori_mask, spar_mask
    """
    # load data
    dataset = VAEDataset(**config["data_params"], pin_memory=len(gpus) != 0)
    dataset.setup("test")
    data_loader = dataset.val_dataloader()

    # true genotype (with MFA imputation)
    geno_true = dataset.val_dataset.genotypes
    # most frequent allele of each SNP
    mf_allele = dataset.val_dataset.mf
    geno_mf = np.tile(np.expand_dims(mf_allele, axis=0), (geno_true.shape[0], 1))

    # original missing mask
    ori_mask = dataset.val_dataset.masks #0 is missing

    mf_errors = []
    AE_errors = []
    miss_rates = []

    for sparsify in sparsifies:
        is_vae = config['model_params']['name'] == "VAE"
        config['model_params']['sparsify']=sparsify
        model = load_model()
        model.train()

        geno_recons = []
        spar_mask =[]
        losses=[]
        with torch.no_grad():
            for geno in tqdm(data_loader):
                geno = geno.to(device)
                if is_vae:
                    recons, input, z, latent_dist, msk = model.forward(geno)
                    loss = model.loss_function(recons, input, z,latent_dist)
                else:
                    recons, input, z, msk = model.forward(geno)
                    loss = model.loss_function(recons, input, z)

                recons = torch.sigmoid(recons)
                geno_recons.append(recons.detach().cpu().numpy())
                spar_mask.append(msk.detach().cpu().numpy())
                losses.append(loss["loss"].detach().cpu())
            print(f"Average loss :{np.array(losses).mean()}")
            geno_recons = np.concatenate(geno_recons,axis=0)
            spar_mask = np.concatenate(spar_mask,axis=0) # 1 is missing
            only_spar_mask = spar_mask*ori_mask
            mse_loss_mf, miss_rate = mse_mask(geno_true,geno_mf,only_spar_mask)
            mse_loss_AE,miss_rate = mse_mask(geno_true,geno_recons,only_spar_mask)
            mf_errors.append(mse_loss_mf)
            AE_errors.append(mse_loss_AE)
            miss_rates.append(miss_rate)
    return mf_errors, AE_errors, miss_rates

if __name__=="__main__":
    inference_all()
    # inference_val_missing()




