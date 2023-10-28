import flax.linen as nn
import jax.numpy as jnp

from .base import Encoder, Aggregator, MLP, Codebook, Decoder, Deaggregator

class MultiHeadVQ(nn.Module):
    """
    MultiHead VQ-VAE for Weather4Cast competition.
    For input and ouput description, look at the definition of the Encoder and Decoder classes.
    """
    config: dict

    @property
    def metrics(self):
        metrics = ['loss', 'recon_loss', 'codebook_loss', 
                   'commitment_loss', 'perplexity', 'codebook_commitment_loss', 'codebook_codebook_loss', 
                   'codebook_perplexity', 'ir_commitment_loss', 'vr_commitment_loss', 'wv_commitment_loss',
                   'ir_codebook_loss', 'vr_codebook_loss', 'wv_codebook_loss',
                   'ir_perplexity', 'wv_perplexity', 'vr_perplexity']
        return metrics

    @nn.compact
    def __call__(self, video, train: bool = True):

        x = video 
        b, t, h, w, c = x.shape
        # Encoder
        x, losses = Encoder(num_blocks=self.config['num_blocks_enc'], filters=self.config['filters_enc'],
                             embeddings=self.config['embeddings_enc'], num_embeddings=self.config['num_embeddings_enc'])(x, train)
        # Data aggregator
        x, queries = Aggregator(embed_dim=self.config['embeddings_enc'], num_heads=self.config['num_heads_agg'],
                                dtype=jnp.float32)(x)

        # Codebook MLP enc
        x = MLP(num_neurons=self.config['mlp_layers'], C=self.config['codebook_embd'])(x)

        # Codebook
        codebook_dict = Codebook(n_codes=self.config['n_codes'], proj_dim=self.config['codebook_embd'],
                                 embedding_dim=self.config['codebook_embd'], dtype=jnp.float32)(x)
        
        x = codebook_dict['embeddings']

        # NOTE: accumulate codebook losses with the other codebooks from the encoder.
        losses['commitment_loss'] += codebook_dict['commitment_loss']
        losses['codebook_loss'] += codebook_dict['codebook_loss']
        losses['perplexity'] += codebook_dict['perplexity']
        losses['codebook_commitment_loss'] = codebook_dict['commitment_loss'] 
        losses['codebook_codebook_loss'] = codebook_dict['codebook_loss']
        losses['codebook_perplexity'] = codebook_dict['perplexity']

        # TODO: add logging
        # TODO: mse losses for the single heads
        # free up space
        del codebook_dict

        # Codebook MLP dec
        x = MLP(num_neurons=self.config['mlp_layers'], C=self.config['embeddings_enc'])(x)

        # Deaggregator
        x = Deaggregator(embed_dim=self.config['embeddings_enc'], num_heads=self.config['num_heads_agg'])(x, queries)
        # free up space
        del queries

        # Decoder
        x = Decoder(num_blocks=self.config['num_blocks_dec'], filters=self.config['filters_dec'],
                    embeddings=self.config['embeddings_enc'], shape=(b * t, 8, 8, self.config['filters_enc'][-1]))(x, train)

        x = x.reshape((b, t, h, w, c))
        return_dict = losses
        return_dict['out'] = x
        return return_dict

        


