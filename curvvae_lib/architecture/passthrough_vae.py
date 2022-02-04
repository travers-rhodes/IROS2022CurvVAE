import torch
from torch import nn, optim
from torch.nn import functional as F
import math

TESTING=True

def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn((mu.shape[0], mu.shape[1]), device=mu.device)
        return mu + eps*std

# take in a nn.linear layer and reset its parameters
# using the standard initialization but then assuming that the 
# in data _WILL_ be scaled by scaled_in and that we want the
# output data _TO BE SCALED_ by scaled_out.
# default params _should_ match default pytorch initialization
def scale_aware_init(linear_layer, in_scale = 1., out_scale = 1.):
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(linear_layer.weight)
    weight_bound = 1/math.sqrt(fan_in) * out_scale / in_scale
    bias_bound = 1/math.sqrt(fan_in) * out_scale
    with torch.no_grad():
        linear_layer.weight.uniform_(-weight_bound, weight_bound)
        linear_layer.bias.uniform_(-bias_bound,bias_bound)

# a Passthrough VAE is a vae which additionally "passes through" a t parameter
# (ie: is forced to store, uncorrupted, a "t" variable which is then returned
# exactly during reconstruction)
class PassthroughVAE(nn.Module):
    def __init__(self):
        super(PassthroughVAE, self).__init__()
    
    def noisy_decode(self, mu, logvar, t):
        z = reparameterize(mu,logvar)
        x, t = self.decode(z, t)
        return(x, z, t)
    
    def noiseless_autoencode(self, x, t):
        mu, logvar,t = self.encode(x, t)
        x,t = self.decode(mu,t)
        return(x,mu,logvar,t)
        
    def forward(self, x, t):
        mu, logvar, t = self.encode(x, t)
        x, noisy_mu, t = self.noisy_decode(mu,logvar,t)
        return(x,mu,logvar, noisy_mu, t)

# input_dimension does not include the passthrough dimension
# emb_layer and recon_layer do not include input/latent dimension sizes
# (and so can be validly set to empty lists)
class FCPassthroughVAE(PassthroughVAE):
  def __init__(self, input_dim, passthrough_dim, latent_dim, 
      emb_layer_widths, recon_layer_widths, dtype, initialization_scale):
    super(FCPassthroughVAE, self).__init__()
    self.input_dim = input_dim
    self.passthrough_dim = passthrough_dim
    self.latent_dim = latent_dim
    self.emb_layer_widths = emb_layer_widths
    self.recon_layer_widths = recon_layer_widths
    self.dtype = dtype
    self.architecture = "fcpassthroughvae"
    self.initialization_scale = initialization_scale

    # initial input has dimension input_dim + passthrough_dim 
    previous_layer_width = self.input_dim + self.passthrough_dim
    self.all_emb_layers = []
    for i, w in enumerate(emb_layer_widths):
      emb_layer = nn.Linear(previous_layer_width, w)
      if i == 0:
        scale_aware_init(emb_layer, in_scale = initialization_scale)
      self.all_emb_layers.append(emb_layer) 
      previous_layer_width = w

    self.fcmu = nn.Linear(previous_layer_width, self.latent_dim)
    self.fclogvar = nn.Linear(previous_layer_width, self.latent_dim)
  
    # create the reconstruction fc layers
    # the output is of size (input_dim) because autoencoding 
    previous_layer_width = self.latent_dim + self.passthrough_dim
    self.all_recon_layers = []
    for w in recon_layer_widths:
      self.all_recon_layers.append(nn.Linear(previous_layer_width, w)) 
      previous_layer_width = w
    
    # add the final layer to get to input_dim (doesn't include passthrough) dimension
    recon_layer = nn.Linear(previous_layer_width, input_dim)
    scale_aware_init(recon_layer, out_scale = initialization_scale)
    self.all_recon_layers.append(recon_layer)

    # set to double as needed
    # maybe equivalent to self=self.double()?
    if dtype == torch.double:
      print("Setting to double")
      for i in range(len(self.all_recon_layers)):
        self.all_recon_layers[i] = self.all_recon_layers[i].double()
      for i in range(len(self.all_emb_layers)):
        self.all_emb_layers[i] = self.all_emb_layers[i].double()
      self.fcmu = self.fcmu.double()
      self.fclogvar = self.fclogvar.double()

    # save layers as ModuleList to record them nicely in module
    self.all_recon_layers = nn.ModuleList(self.all_recon_layers)
    self.all_emb_layers = nn.ModuleList(self.all_emb_layers)
    
  # x is expected to be a Tensor of the form
  # batchsize x (spatialdims + rotdims)
  # t is expected to be a Tensor of the form
  # batchsize
  # and the output is a pair of Tensors of sizes
  # batchsize x self.latent_dim
  # and
  # batchsize x self.latent_dim
  def encode(self, x, t):
    if TESTING:
      assert len(x.shape) == 2, "batch of x coords should be two dimensional"
      assert len(t.shape) == 2, "batch of t vals should be two dimensional"
      assert x.shape[0] == t.shape[0], "inputs should have same batchsize"
    layer = torch.hstack((x,t))
    for fc in self.all_emb_layers:
        layer = F.relu(fc(layer))
    mu = self.fcmu(layer)
    logvar = self.fclogvar(layer)
    return(mu,logvar,t)
    
  def decode(self, z, t):
    if TESTING:
      assert len(z.shape) == 2, "batch of z coords should be two dimensional"
      assert len(t.shape) == 2, "batch of t vals should be two dimensional"
      assert z.shape[0] == t.shape[0], "inputs should have same batchsize"
    layer = torch.hstack((z,t))
    num_fcs = len(self.all_recon_layers)
    for i, fc in enumerate(self.all_recon_layers):
      layer = fc(layer)
      # add relu on every layer except the last one
      if i != num_fcs-1:
        layer = F.relu(layer)
    return(layer,t)
