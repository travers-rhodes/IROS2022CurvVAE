import pickle
import torch

from curvvae_lib.architecture.passthrough_vae import FCPassthroughVAE 

def load_model(model_folder_path, device="cpu"):
    with open(model_folder_path + "/model_type.txt", "r") as f:
      model_type = f.readline().strip()
    with open(model_folder_path + "/model_args.p", "rb") as f:
      kwargs = pickle.load(f)
    if model_type == "fcpassthroughvae":
        model = FCPassthroughVAE(**kwargs).to(device)
    else:
        raise ValueError("The model type (%s) was not a known type" % model_type)

    model.load_state_dict(torch.load(model_folder_path + "/model_state_dict.pt", map_location = torch.device(device)))
    return model 
