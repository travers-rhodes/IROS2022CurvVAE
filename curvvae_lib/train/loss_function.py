import torch

TESTING=True # Set to True to run a bunch of extra asserts

# Reconstruction + KL divergence losses summed over all pixels and batch
def vae_loss_function(recon_x, x, mu, logvar, beta, loss_func="binary_cross_entropy"):
    # To make the units work properly, this should be equal to
    # the log reconstruction probability, which is
    # log[N(x, F(z), sigma^2_Recon (assume to be 1 for simplicity))]
    # We can ignore the constant term, since that won't change our optimization objective.
    # but we still get a factor of 0.5 here we were missing before

    # confirm that the first dimension of every input tensor is batch_size
    batch_size = recon_x.shape[0]
    if TESTING:
        assert x.shape[0] == batch_size, "x should have batch_size as first dimension"
        assert mu.shape[0] == batch_size, "mu should have batch_size as first dimension"
        assert logvar.shape[0] == batch_size, "logvar should have batch_size as first dimension"

    noiselessLogLikelihood = torch.tensor(0)
    position_loss = torch.tensor(0)
    quat_loss = torch.tensor(0)
    if loss_func == "binary_cross_entropy": 
        # recon_x is of shape batchsize x im_channels x im_side_len x im_side_len
        LogLikelihood = - torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')/batch_size
    elif loss_func == "gaussian":
        # recon_x is of shape batchsize x im_channels x im_side_len x im_side_len
        LogLikelihood = - torch.nn.functional.mse_loss(recon_x, x, reduction='sum')/batch_size
    elif loss_func == "pose_loss":
        # recon_x is of shape batchsize x 7 x traj_len 
        total_loss, position_loss, quat_loss = pose_reconstruction_loss(recon_x, x)
        position_loss = position_loss/batch_size
        quat_loss = quat_loss/batch_size
        LogLikelihood = - total_loss/batch_size
    else:
        print(f"Unknown loss function: {loss_func}")
        

    if TESTING:
        assert len(LogLikelihood.shape)==0, "LogLikelihood should be scalar"
  
    ########### mu is of shape batchsize x latent_dim
    mu_error = -0.5 * torch.sum(- mu.pow(2))/batch_size
    if TESTING:
        assert len(mu_error.shape)==0,"mu_error should be scalar"

    ########### mu is of shape batchsize x latent_dim
    # note that the 1 is getting broadcast batchsize x latent_dim times 
    # (so this calc correctly implements Appendix B of Auto-Encoding Variational Bayes)
    logvar_error = -0.5 * torch.sum(1 + logvar - logvar.exp())/batch_size
    if TESTING:
        assert len(logvar_error.shape)==0,"logvar_error should be scalar"
  
    #print("kl content", latent_dim + logvar - mu.pow(2) - logvar.exp())
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = logvar_error + mu_error
    if TESTING:
        assert len(KLD.shape)==0,"KLD should be scalar"
  
    # LogLikelihood - KLD is a lower bound
    # We want to maximize that lower bound
    # So, we take the negative of our lower bound to get the ELBO "cost"
    return (-LogLikelihood + beta * KLD, -LogLikelihood, KLD, mu_error, logvar_error, position_loss, quat_loss)
