import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from .base import Encoder, Aggregator, MLP, Codebook, Decoder, Deaggregator, EncoderHead, OneHeadEncoder, OneHeadDecoder


class OneHeadEncDec(nn.Module):
    """
    Encoder-MLP-Codebook-MLP-Decoder.
    """
    config: dict

    @property
    def metrics(self):
        metrics = ['loss', 'recon_loss', 'codebook_loss', 
                   'commitment_loss', 'perplexity', 
                   'usage']
        return metrics
    
    @nn.compact
    def __call__(self, video, train: bool):
        x = video
        b, t, h, w, c = x.shape
        # Encoder
        x = OneHeadEncoder(num_blocks=self.config['num_blocks_enc'], filters=self.config['filters_enc'],
                             embeddings=self.config['embeddings_enc'])(x, train)

        # Collapse time and batch dimensions
        x = x.reshape(-1, x.shape[-1])
        x = MLP(num_neurons=self.config['mlp_layers'], C=self.config['codebook_embd'])(x)
        # Codebook
        codebook_dict = Codebook(n_codes=self.config['n_codes'], proj_dim=self.config['codebook_embd'],
                                 embedding_dim=self.config['codebook_embd'], dtype=jnp.float32)(x)
        
        x = codebook_dict['embeddings']

        x = MLP(num_neurons=self.config['mlp_layers'], C=self.config['embeddings_enc'])(x)

        x = x.reshape(b, t, -1)
        x = OneHeadDecoder(num_blocks=self.config['num_blocks_dec'], filters=self.config['filters_dec'],
                    embeddings=self.config['embeddings_enc'], shape=(b * t, 8, 8, self.config['filters_enc'][-1]))(x, train)
        
        x = x.reshape((b, t, h, w, c))
        return_dict = dict(codebook_loss=codebook_dict['codebook_loss'], commitment_loss=codebook_dict['codebook_loss'],
                           perplexity=codebook_dict['perplexity'])
        return_dict['out'] = x
        #return_dict.update(usage)
        return_dict['usage'] = codebook_dict['encodings']
        return return_dict
        
