import torch.utils.data
from torch.utils.data.dataloader import default_collate

from batch import BatchSubstructContext, BatchMasking, BatchAE

class DataLoaderSubstructContext(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchSubstructContext.from_data_list(data_list),
            **kwargs)

class DataLoaderMasking(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMasking, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMasking.from_data_list(data_list),
            **kwargs)


from util import MaskAtom
class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        # self._transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = mask_rate, mask_edge=mask_edge)
        self._transform = MaskAtom(num_atom_type = 119, num_edge_type = 23, mask_rate = mask_rate, mask_edge=mask_edge)
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)


########## [3D-level] for 3D-level pretext task ##########
from util import Dist3d
class DataLoaderDist3DPred(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        self._transform = Dist3d()
        super(DataLoaderDist3DPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)
########## [3D-level] for 3D-level pretext task ##########


########## [MORE] Random Masking + 3D Distance ##########
from util import MaskAtom3dDist
class DataLoaderMasking3dDistPred(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        self._transform = MaskAtom3dDist(num_atom_type = 119, num_edge_type = 23, mask_rate = mask_rate, mask_edge=mask_edge)
        super(DataLoaderMasking3dDistPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)
########## [MORE] Random Masking + 3D Distance ##########


class DataLoaderAE(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How may samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`True`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderAE, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchAE.from_data_list(data_list),
            **kwargs)



