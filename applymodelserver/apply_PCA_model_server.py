import http.server
import socketserver
import numpy as np

import sys
sys.path.append("../")



# should be T x latent_dim_size vector
pcamodel = np.load(f"../notebooks/pcamodels/banana_lat3_pca_20220725-203021.npz")
loaded_pca_components = pcamodel["pca_components"]
loaded_pca_means = pcamodel["mean"]

# a bit hacky, but for now assert the latent_dim is 3
# since that will make re-using the VAE version of this API easier
latent_dim = 3
np.testing.assert_equal(loaded_pca_components.shape[1], latent_dim)


PORT = 8092

class Handler(http.server.BaseHTTPRequestHandler):
  # https://gist.github.com/mdonkers/63e115cc0c79b4f6b8b3a6b797e485c7
  def do_POST(self):
    content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
    post_data = self.rfile.read(content_length) # <--- Gets the data itself
    # assume that the passthrough_dim is 1
    # and that the structure of the passed data is (passthrough_dim1, passthrough_dim2,..., lat_dim1, lat_dim2, ...)
    latent_vector = np.frombuffer(post_data, dtype=np.float32)
    # SUPER HACKY here, but assert that latent vector is of length 3
    # and that way we can assume that this vector is of length (64 timesteps x (timeindex + 3 latent values))
    latent_vector = latent_vector.reshape(-1,1 + latent_dim)[0,1:] 
    latent_vector = latent_vector.reshape(3,1)
    print(latent_vector)
    response_x = loaded_pca_components @ latent_vector 
    response = response_x + loaded_pca_means.reshape(-1,1) 
    response = response.astype(np.float32)

    #https://wiki.python.org/moin/BaseHttpServer 
    self.send_response(200)
    self.send_header("Content-type", "numpy-array-in-bytes")
    self.end_headers()
    self.wfile.write(response.tobytes())

  

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
