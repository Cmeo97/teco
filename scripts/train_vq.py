import os
import os.path as osp
import numpy as np
import time
import argparse
import yaml
import pickle
import wandb
import glob

import optax
import jax
from jax import random
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

from teco.data import Data
from teco.train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all
from teco.utils import flatten, add_border, save_video_grid
from teco.models import get_model, sample
from teco.data import load_weather4cast
from teco.w4c_data_utils import load_config
from teco.models.onehead import OneHeadEncDec


def main():
    global model
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    files = glob.glob(osp.join(config.output_dir, 'checkpoints', '*'))
    if len(files) > 0:
        print('Found previous checkpoints', files)
        config.ckpt = config.output_dir
    else:
        config.ckpt = None

    if is_master_process:
        root_dir = os.environ['DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)

        wandb.init(project='teco', config=config,
                   dir=root_dir, id='gamma-encdec')
        wandb.run.name = 'gamma-encdec'
        wandb.run.save()

    #data = Data(config)
    #train_loader = data.create_iterator(train=True)
    #test_loader = data.create_iterator(train=False)

    train_data = load_weather4cast(config, 'train')
    test_data = load_weather4cast(config, 'validation')

    num_data_local = jax.local_device_count() # number of GPUs used (has to be divisible by the batch size)
    def prepare_tf_data(xs):
            def _prepare(x):
                if x == None:
                    return None
                x = x._numpy()
                x = x.reshape((num_data_local, -1) + x.shape[1:])
                return x
            xs = jax.tree_util.tree_map(_prepare, xs)
            return xs

    # train
    iterator = map(prepare_tf_data, train_data)
    iterator = jax_utils.prefetch_to_device(iterator, 2)
    train_loader = iterator
    # validation
    iterator = map(prepare_tf_data, test_data)
    iterator = jax_utils.prefetch_to_device(iterator, 2)
    test_loader = iterator
    batch = next(train_loader)
    batch = get_first_device(batch)
    model = get_model(config)
    state, schedule_fn = init_model_state(init_rng, model, batch, config, train_flag=True)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    rngs = random.split(rng, jax.local_device_count())
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, model, state, train_loader,
                                       test_loader, schedule_fn, rngs)
        if iteration % config.save_interval == 0:
            if is_master_process:
                state_ = jax_utils.unreplicate(state)
                save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)
                print('Saved checkpoint to', save_path)
                del state_ # Needed to prevent a memory leak bug
        iteration += 1


def train_step(batch, state, rng):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        out, updates = state.apply_fn(
            variables,
            video=batch['video'],
            train=True,
            rngs=rngs,
            mutable=['batch_stats']
        )
        # mean squared error loss function
        recon_loss = optax.squared_error(out['out'], batch['video']).mean()
        # final loss is determined by the sum of reconstruction, commitment and codebook loss
        loss = recon_loss + out['codebook_loss'] + out['commitment_loss']
        out['recon_loss'] = recon_loss
        out['loss'] = loss
        return loss, (out, updates)

    aux, grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    out = aux[1][0]
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(
        grads=grads,
    )
    # apply batch stats
    state = state.replace(batch_stats=aux[1][1]['batch_stats'])

    return state, out, new_rng

def eval_step(batch, state, rng):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}
    variables = {'params': state.params, 'batch_stats': state.batch_stats.unfreeze()}
    out = state.apply_fn(
        variables,
        video=batch['video'],
        train=False,
        rngs=rngs,
    )

    recon_loss = optax.squared_error(out['out'], batch['video']).mean()
    loss = recon_loss + out['codebook_loss'] + out['commitment_loss']
    out['recon_loss'] = recon_loss
    out['loss'] = loss

    return state, out, new_rng


def train(iteration, model, state, train_loader, test_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    p_train_step = jax.pmap(train_step, axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    # Load model configuration.
    model_config = load_config(config.model_configuration_path)
    end = time.time()
    metrics = {k: jnp.array([0]) for k in model.metrics}
    metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
    while True:
        batch = next(train_loader)
        batch_size = batch['video'].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        if isinstance(model, OneHeadEncDec):
            return_dict['usage'] = np.array([len(np.unique(return_dict['usage'])) / model_config['n_codes']]) 
        else:
            # The percentage usage is average over all samples in a batch
            return_dict['ir_usage'] = np.array([len(np.unique(return_dict['ir_usage'])) / model_config['num_embeddings_enc']])
            return_dict['vr_usage'] = np.array([len(np.unique(return_dict['vr_usage'])) / model_config['num_embeddings_enc']])
            return_dict['wv_usage'] = np.array([len(np.unique(return_dict['wv_usage'])) / model_config['num_embeddings_enc']])

        metrics = {k: return_dict[k].mean() + metrics[k] for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            # Average over iterations.
            if iteration == 0:
                metrics = {k: metrics[k] for k in model.metrics} 
            else:
                metrics = {k: metrics[k] / config.log_interval for k in model.metrics} 

            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

            # Reset metrics after logging.            
            metrics = {k: jnp.array([0]) for k in model.metrics}
            metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()} 
            
            # VALIDATION
            val_metrics = {k: jnp.array([0]) for k in model.metrics}
            val_metrics = {k: v.astype(jnp.float32) for k, v in val_metrics.items()}
            val_iterations = 0
            eval_batch = next(test_loader)
            iterations = 1 if iteration == 0 else config.log_interval
            for _ in range(iterations): # NOTE: use the same number of iterations for validation as for training.
                batch_size = eval_batch['video'].shape[1]

                state, return_dict, rngs = p_eval_step(batch=eval_batch, state=state, rng=rngs)
                if isinstance(model, OneHeadEncDec):
                    return_dict['usage'] = np.array([len(np.unique(return_dict['usage'])) / model_config['n_codes']]) 
                else:
                    # The percentage usage is average over all samples in a batch
                    return_dict['ir_usage'] = np.array([len(np.unique(return_dict['ir_usage'])) / model_config['num_embeddings_enc']])
                    return_dict['vr_usage'] = np.array([len(np.unique(return_dict['vr_usage'])) / model_config['num_embeddings_enc']])
                    return_dict['wv_usage'] = np.array([len(np.unique(return_dict['wv_usage'])) / model_config['num_embeddings_enc']])
                val_metrics = {k: (return_dict[k].mean() + val_metrics[k]) for k in model.metrics}
                val_iterations += 1
                eval_batch = next(test_loader)

            val_metrics = {k: val_metrics[k] / val_iterations for k in model.metrics}
            print(val_metrics, val_iterations)
            # LOG VALIDATION
            wandb.log({**{f'val/{metric}': val
                        for metric, val in val_metrics.items()}
                    }, step=iteration)


        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            #progress.display(iteration)
            print(f'Logged iteration {iteration}')

        if iteration % config.save_interval == 0 or \
        iteration % config.viz_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs

        iteration += 1


def visualize(model, iteration, state, test_loader):
    batch = next(test_loader)

    predictions, real = sample(model, state, batch['video'], batch['actions'])
    predictions, real = jax.device_get(predictions), jax.device_get(real)
    predictions, real = predictions * 0.5 + 0.5, real * 0.5 + 0.5
    predictions = flatten(predictions, 0, 2)
    add_border(predictions[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(predictions[:, config.open_loop_ctx:], (1., 0., 0.))

    original = flatten(real, 0, 2)
    video = np.stack((predictions, original), axis=1) # (NB)2THWC
    video = flatten(video, 0, 2) # (NB2)THWC
    video = save_video_grid(video)
    video = np.transpose(video, (0, 3, 1, 2))
    if is_master_process:
        wandb.log({'viz/sample': wandb.Video(video, fps=20, format='gif')}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            os.environ['DATA_DIR'] = 'logs'
            print('DATA_DIR environment variable not set, default to logs/')
        root_folder = os.environ['DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['save_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    main()
