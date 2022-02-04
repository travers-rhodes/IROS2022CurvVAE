import torch
import torch.utils.tensorboard
import os
import numpy as np
import curvvae_lib.architecture.passthrough_vae as pv 
from curvvae_lib.train.loss_function import vae_loss_function 

def detached_length(vector):
    all_dims_but_first = list(range(len(vector.shape)))[1:]
    new_shape = np.ones(len(vector.shape))
    new_shape[0] = -1
    new_shape = list(new_shape.astype(int))
    length = torch.sqrt(torch.sum(torch.square(vector), dim=all_dims_but_first)).detach().reshape(new_shape)
    return length

def curvature_estimate(model, zvalues, tvalues, device, epsilon_scale = 0.001, epsilon_div_zero_fix = 1e-10):
    # 2021-12-15: turns out that torch.normal can return exactly zero. fun fact.
    # resampling is the easy/inefficient fix
    computed_valid_hvalues = False
    while not computed_valid_hvalues:
        hvalues_raw = torch.normal(torch.zeros(size=zvalues.shape, device=device), 
                                   torch.ones(size=zvalues.shape, device=device))   
        hvalues_length = torch.sqrt(torch.sum(torch.square(hvalues_raw), axis=1)).reshape(-1,1)  
        hvalues = hvalues_raw/hvalues_length * epsilon_scale
        computed_valid_hvalues = not torch.any(hvalues.isnan())
    zvalues_detached = torch.tensor(zvalues.detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_plus_hvalues_detached = torch.tensor((zvalues+hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_minus_hvalues_detached = torch.tensor((zvalues-hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    #print(zvalues.shape, hvalues.shape, hvalues_length.shape)    
    modeled_z, t = model.decode(zvalues_detached, tvalues)
    modeled_z_plus_h, t = model.decode(z_plus_hvalues_detached, tvalues)
    modeled_z_minus_h, t = model.decode(z_minus_hvalues_detached, tvalues)
    
    forward_length = detached_length(modeled_z_plus_h - modeled_z) + epsilon_div_zero_fix
    backward_length = detached_length(modeled_z - modeled_z_minus_h) + epsilon_div_zero_fix
    #print(modeled_z.shape, modeled_z_plus_h.shape, forward_length.shape)
    curv_est = ((modeled_z_plus_h - modeled_z)/forward_length - (modeled_z - modeled_z_minus_h)/backward_length)/forward_length
    #print(second_est)
    error = torch.sum(torch.square(curv_est))/curv_est.shape[0]
    #return(error, second_est, hvalues)
    return error

def second_deriv_estimate(model, zvalues, tvalues, device, epsilon_scale = 0.001):
    # 2021-12-15: turns out that torch.normal can return exactly zero. fun fact.
    # resampling is the easy/inefficient fix
    computed_valid_hvalues = False
    while not computed_valid_hvalues:
        hvalues_raw = torch.normal(torch.zeros(size=zvalues.shape, device=device), 
                                   torch.ones(size=zvalues.shape, device=device))   
        hvalues_length = torch.sqrt(torch.sum(torch.square(hvalues_raw), axis=1)).reshape(-1,1)  
        hvalues = hvalues_raw/hvalues_length * epsilon_scale
        computed_valid_hvalues = not torch.any(hvalues.isnan())
    zvalues_detached = torch.tensor(zvalues.detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_plus_hvalues_detached = torch.tensor((zvalues+hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_minus_hvalues_detached = torch.tensor((zvalues-hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    #print(zvalues.shape, hvalues.shape, hvalues_length.shape)    
    modeled_z,t = model.decode(zvalues_detached, tvalues)
    modeled_z_plus_h,t = model.decode(z_plus_hvalues_detached, tvalues)
    modeled_z_minus_h,t = model.decode(z_minus_hvalues_detached, tvalues)
    second_est = (modeled_z_plus_h + modeled_z_minus_h - 2 * modeled_z)/(epsilon_scale**2)
    #print(second_est)
    error = torch.sqrt(torch.sum(torch.square(second_est))/second_est.shape[0])
    #return(error, second_est, hvalues)
    return error#, hvalues_raw


# The PredictivePassThrough should take in a _PAIR_ of data from the same trajectory datax1,datat1 datax2,datat2
# and should then flow as datax1,datat1 -> z1,t1 -> z1,t2 -> reconx2,recont2 with loss comparing reconx2 with datax2
class PPTVAETrainer(object):
    def __init__(self, model, data_loader, beta, device, 
        log_dir, lr, annealingBatches, record_loss_every=100, loss_func="binary_cross_entropy"):
      self.model = model
      self.data_loader = data_loader
      self.beta = beta
      self.optimizer= torch.optim.Adam(self.model.parameters(), lr=lr)
      self.device = device
      self.log_dir = log_dir
      self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=self.log_dir)
      self.num_batches_seen = 0
      self.annealingBatches = annealingBatches
      self.record_loss_every = record_loss_every
      self.loss_func = loss_func

    def train(self, second_deriv_regularizer=0, curvature_regularizer=0, num_new_samp_points=None, epsilon_scale = 0.001):
        
      # set model to "training mode"
      self.model.train()

      # previous step model
      prevmodel = pv.FCPassthroughVAE(self.model.input_dim,
          self.model.passthrough_dim, self.model.latent_dim, self.model.emb_layer_widths,
          self.model.recon_layer_widths, self.model.dtype, self.model.initialization_scale)


      # for each chunk of data in an epoch
      for datax1,datat1,datax2,datat2 in self.data_loader:
        # do annealing and store annealed value to tmp value
        if self.num_batches_seen < self.annealingBatches:
          tmp_beta = self.beta * self.num_batches_seen / self.annealingBatches
        else:
          tmp_beta = self.beta

        # move data to device, initialize optimizer, model data, compute loss,
        # and perform one optimizer step
        x = datax1.to(self.device)
        t = datat1.to(self.device)
        x2 = datax2.to(self.device)
        t2 = datat2.to(self.device)
        self.optimizer.zero_grad()
        mu, logvar, t = self.model.encode(x, t)
        noisy_mu = pv.reparameterize(mu,logvar)
        reconx2, t2 = self.model.decode(noisy_mu,t2)
        loss, NegLogLikelihood, KLD, mu_error, logvar_error, position_loss, quat_loss = vae_loss_function(reconx2, 
                                      x2, mu, logvar, tmp_beta, self.loss_func)
     
        if num_new_samp_points is not None:
            samp_embs = torch.hstack((noisy_mu,t))
            samp_mean = torch.mean(samp_embs.T, axis=1).detach().cpu().numpy()
            singularity_removal_epsilon = 1e-8
            samp_cov = np.cov(samp_embs.T.detach().cpu().numpy())
            if np.any(np.isnan(samp_mean)) or np.any(np.isinf(samp_mean)):
                print(samp_mean)
                print(samp_embs)
                print(samp_cov)
            if np.any(np.isnan(samp_cov)) or np.any(np.isinf(samp_cov)):
                print(samp_mean)
                print(samp_embs)
                print(samp_cov)
            if samp_cov.size==1:
                samp_cov = samp_cov.reshape((1,1))
            samp_points = torch.tensor(np.random.multivariate_normal(mean=samp_mean, cov=samp_cov,size=num_new_samp_points), dtype=self.model.dtype).to(self.device)
         
        if second_deriv_regularizer == 0:
            second_deriv_loss = torch.tensor(0)
        elif num_new_samp_points is None:
            second_deriv_loss = second_deriv_estimate(self.model, noisy_mu, t, self.device, epsilon_scale = epsilon_scale)
        else:
            second_deriv_loss = second_deriv_estimate(self.model, samp_points[:,:self.model.latent_dim], samp_points[:,self.model.latent_dim:], self.device, epsilon_scale = epsilon_scale)
        
        if curvature_regularizer == 0:
            curvature_loss = torch.tensor(0)
        elif num_new_samp_points is None:
            curvature_loss = curvature_estimate(self.model, noisy_mu, t, self.device, epsilon_scale = epsilon_scale)
        else:
            curvature_loss = curvature_estimate(self.model, samp_points[:,:self.model.latent_dim], samp_points[:,self.model.latent_dim:], self.device, epsilon_scale = epsilon_scale)

        loss = loss + second_deriv_loss * second_deriv_regularizer + curvature_loss * curvature_regularizer
        
        loss.backward()
        prevmodel.load_state_dict(self.model.state_dict())
        self.optimizer.step()
        # https://discuss.pytorch.org/t/network-parameters-becoming-inf-after-first-optimization-step/106137
        for key, val in self.model.state_dict().items():
          if val.isnan().any():
            print(f'Found nan values in network element {key}:\n{val}')
            return True, self.model, prevmodel, datax1,datat1,datax2,datat2,noisy_mu
          if val.isinf().any():
            print(f'Found inf values in network element {key}:\n{val}')
            return True, self.model, prevmodel, datax1,datat1,datax2,datat2,noisy_mu

        # log to tensorboard
        self.num_batches_seen += 1
        if self.num_batches_seen % self.record_loss_every == 0:
            self.writer.add_scalar("ELBO/train", loss.item(), self.num_batches_seen) 
            self.writer.add_scalar("KLD/train", KLD.item(), self.num_batches_seen) 
            self.writer.add_scalar("MuDiv/train", mu_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("VarDiv/train", logvar_error.item(), self.num_batches_seen) 
            self.writer.add_scalar("NLL/train", NegLogLikelihood.item(), self.num_batches_seen) 
            self.writer.add_scalar("beta", tmp_beta, self.num_batches_seen) 
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.num_batches_seen)
            self.writer.add_scalar("pos_loss", position_loss.item(), self.num_batches_seen)
            self.writer.add_scalar("quat_loss", quat_loss.item(), self.num_batches_seen)
            self.writer.add_scalar("second_deriv_loss", second_deriv_loss.item(), self.num_batches_seen)
            self.writer.add_scalar("curvature_loss", curvature_loss.item(), self.num_batches_seen)
            self.writer.add_scalar("finite_diff_step", epsilon_scale, self.num_batches_seen)
      return False, self.model, prevmodel, datax1,datat1,datax2,datat2,noisy_mu
