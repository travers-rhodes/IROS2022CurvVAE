import http.server
import socketserver
import numpy as np

import sys
sys.path.append("../")
import torch
from torch.nn import functional as F
import curvvae_lib.architecture.load_model as lm



loaded_vae = lm.load_model(
f"../notebooks/trainedmodels/banana_lat3_curvreg0.001_beta0.001_20220209-120436"
#f"../notebooks/trainedmodels/carrot_lat3_curvreg0.001_beta0.001_20220216-113851"
,"cpu")


PORT = 8090

class Handler(http.server.BaseHTTPRequestHandler):
  # https://gist.github.com/mdonkers/63e115cc0c79b4f6b8b3a6b797e485c7
  def do_POST(self):
    content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
    post_data = self.rfile.read(content_length) # <--- Gets the data itself
    # assume that the passthrough_dim is 1
    # and that the structure of the passed data is (passthrough_dim1, passthrough_dim2,..., lat_dim1, lat_dim2, ...)
    latent_vector = np.frombuffer(post_data, dtype=np.float32)
    latent_vector = latent_vector.reshape(-1,loaded_vae.passthrough_dim + loaded_vae.latent_dim)
    print(latent_vector)
    t = torch.tensor(latent_vector[:,:loaded_vae.passthrough_dim], dtype=torch.float)
    z = torch.tensor(latent_vector[:,loaded_vae.passthrough_dim:], dtype=torch.float)
    print(t,z)
    response_x, response_t = loaded_vae.decode(z,t)
    response = response_x.detach().cpu().numpy()
    response.astype(np.float32)

    #https://wiki.python.org/moin/BaseHttpServer 
    self.send_response(200)
    self.send_header("Content-type", "numpy-array-in-bytes")
    self.end_headers()
    self.wfile.write(response.tobytes())

  

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
