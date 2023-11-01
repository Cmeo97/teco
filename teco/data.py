import glob
import os.path as osp
import numpy as np
from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
from tensorflow.python.lib.io import file_io
import io
import torch

from teco.w4c_data import RainData
from teco.w4c_data_utils import load_config


def is_tfds_folder(path):
    path = osp.join(path, '1.0.0')
    if path.startswith('gs://'):
        return tf.io.gfile.exists(path)
    else:
        return osp.exists(path)
    
class Generator:

    def __init__(self, data, epochs: int, num_samples: int, mode: str):
        self.data = data
        self.epochs = epochs
        self.num_samples = num_samples
        self.mode = mode
        if self.num_samples == -1:
            self.num_samples = len(self.data)

    def data_generator(self):
        # NOTE: Use this instead the number of epochs as the repo setup already has a way to control training length
        while True:
            ids = np.random.permutation(np.arange(len(self.data)))
            # shuffle dataset
            self.data = torch.utils.data.Subset(self.data, ids)

            for j in range(self.num_samples):
                x, y, z = self.data[j]
                # Shape of data: (C x T x W x H)
                # Change the shape to (T x H x W x C)
                x = np.transpose(x, (1, 3, 2, 0))
                y = np.transpose(y, (1, 3, 2, 0))
                x = np.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)))
                y = np.pad(y, ((0, 0), (2, 2), (2, 2), (0, 0)))
                sample = np.vstack((x, y))
                actions = np.zeros(sample.shape[0]).astype(np.int32)

                yield sample, actions


    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(self.data_generator,
            output_signature=(tf.TensorSpec(shape=(36, 256, 256, 11), dtype=np.float32),
                tf.TensorSpec(shape=(36), dtype=np.int32)))
        return dataset


def load_weather4cast(config: dict, split: str):
    """
    This function returns a Pytorch dataset.
    """
    # YAML file specific for loading the Weather4Cast data.
    dataset_config_path = config.data_configuration_path
    dataset_config = load_config(dataset_config_path)
    if split == 'train':
        dataset = RainData('training', **dataset_config['dataset'])
    else:
        dataset = RainData('validation', **dataset_config['dataset'])
    generator = Generator(dataset, config.epochs, config.num_samples, split)
    dataset = generator.get_dataset()
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(config.batch_size)
    return dataset

def load_npz(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.npz')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        data = np.load(path)
        video, actions = data['video'].astype(np.float32), data['actions'].astype(np.int32)
        video = 2 * (video / 255.) - 1 
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset


def load_video(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.mp4')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    # TODO resizing video
    def read(path):
        path = path.decode('utf-8')

        video = tfio.experimental.ffmpeg.decode_video(tf.io.read_file(path)).numpy()
        start_idx = np.random.randint(0, video.shape[0] - config.seq_len + 1)
        video = video[start_idx:start_idx + config.seq_len]
        video = 2 * (video / np.array(255., dtype=np.float32)) - 1
        
        np_path = path[:-3] + 'npz'
        if tf.io.gfile.exists(np_path):
            if path.startswith('gs://'):
                np_path = io.BytesIO(file_io.FileIO(np_path, 'rb').read())
            np_data = np.load(np_path)
            actions = np_data['actions'].astype(np.int32)
            actions = actions[start_idx:start_idx + config.seq_len]
        else:
            actions = np.zeros((video.shape[0],), dtype=np.int32)
        
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset
                

class Data:
    def __init__(self, config, xmap=False):
        self.config = config
        self.xmap = xmap
        print('Dataset:', config.data_path)

    @property
    def train_itr_per_epoch(self):
        return self.train_size // self.config.batch_size

    @property
    def test_itr_per_epoch(self):
        return self.test_size // self.config.batch_size

    def create_iterator(self, train, repeat=True, prefetch=True):
        if self.xmap:
            num_data = jax.device_count() // self.config.num_shards
            num_data_local = max(1, jax.local_device_count() // self.config.num_shards)
            if num_data >= jax.process_count():
                num_ds_shards = jax.process_count()
                ds_shard_id = jax.process_index()
            else:
                num_ds_shards = num_data
                n_proc_per_shard = jax.process_count() // num_data
                ds_shard_id = jax.process_index() // n_proc_per_shard
        else:
            num_data_local = jax.local_device_count()
            num_ds_shards = jax.process_count()
            ds_shard_id = jax.process_index()

        batch_size = self.config.batch_size // num_ds_shards
        split_name = 'train' if train else 'test'

        if not is_tfds_folder(self.config.data_path):
            if 'dmlab' in self.config.data_path:
                dataset = load_npz(self.config, split_name, num_ds_shards, ds_shard_id)
            else:
                dataset = load_video(self.config, split_name, num_ds_shards, ds_shard_id)
        else:
            seq_len = self.config.seq_len
            def process(features):
                video = tf.cast(features['video'], tf.int32)
                T = tf.shape(video)[0]
                start_idx = tf.random.uniform((), 0, T - seq_len + 1, dtype=tf.int32)
                video = tf.identity(video[start_idx:start_idx + seq_len])
                actions = tf.cast(features['actions'], tf.int32)
                actions = tf.identity(actions[start_idx:start_idx + seq_len])
                return dict(video=video, actions=actions)

            split = tfds.split_for_jax_process(split_name, process_index=ds_shard_id,
                                               process_count=num_ds_shards)
            dataset = tfds.load(osp.basename(self.config.data_path), split=split,
                                data_dir=osp.dirname(self.config.data_path))

            # caching only for pre-encoded since raw video will probably
            # run OOM on RAM
            if self.config.cache:
                dataset = dataset.cache()

            options = tf.data.Options()
            options.threading.private_threadpool_size = 48
            options.threading.max_intra_op_parallelism = 1
            dataset = dataset.with_options(options)
            dataset = dataset.map(process)

        if repeat:
            dataset = dataset.repeat()
        if train:
            dataset = dataset.shuffle(batch_size * 32, seed=self.config.seed)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)

        def prepare_tf_data(xs):
            def _prepare(x):
                x = x._numpy()
                x = x.reshape((num_data_local, -1) + x.shape[1:])
                return x
            xs = jax.tree_util.tree_map(_prepare, xs)
            return xs

        iterator = map(prepare_tf_data, dataset)

        if prefetch:
            iterator = jax_utils.prefetch_to_device(iterator, 2)

        return iterator
