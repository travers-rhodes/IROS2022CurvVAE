import pickle
import os
import torch

# Save the model in an custom-code-readable way
def save_fcpassthrough_vae(vae, model_folder_path):
    kwargs = {
              "input_dim": vae.input_dim,
              "passthrough_dim": vae.passthrough_dim,
              "latent_dim": vae.latent_dim,
              "emb_layer_widths": vae.emb_layer_widths,
              "recon_layer_widths": vae.recon_layer_widths,
              "dtype": vae.dtype
             }
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    with open(model_folder_path + "/model_args.p", "wb") as f:
        pickle.dump(kwargs, f)
    with open(model_folder_path + "/model_type.txt", "w") as f:
        f.write(vae.architecture)
    torch.save(vae.state_dict(), model_folder_path + "/model_state_dict.pt")
