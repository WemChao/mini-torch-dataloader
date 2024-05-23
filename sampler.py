import itertools
import numpy as np
from torch.utils.data.sampler import Sampler
from utils import comm


class InferenceSampler(Sampler):
    """
    Produce indices for inference across all workers.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self, repeat_factors, *, shuffle=True, seed=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.cur_rank
        self._world_size = comm.get_world_size()
        self._int_part = np.trunc(repeat_factors).astype(np.int32)
        self._frac_part = repeat_factors - self._int_part


    def __len__(self):
        return len(self._int_part)
    
    def _get_epoch_indices(self, generator: np.random.Generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (paddle.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            paddle.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training

        rands = generator.random(len(self._frac_part))
        rep_factors = self._int_part + (rands < self._frac_part).astype(np.int32)
        # Construct a list of indices in which we repeat images as specified
        indices = np.repeat(np.arange(len(rep_factors)), rep_factors)
        return indices

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        rng = np.random.default_rng(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(rng)
            if self._shuffle:
                randperm = rng.permutation(len(indices))
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()


