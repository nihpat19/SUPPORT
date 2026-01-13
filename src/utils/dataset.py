import skimage.io as skio
import numpy as np
import torch
import zarr
import pdb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
try:
    from SUPPORT.src.utils.util import get_coordinate
except ModuleNotFoundError:
    from src.utils.util import get_coordinate
import pipeline
from pipeline.utils import galvo_corrections
import tifffile
from pipeline.experiment import Scan
import pipeline.reso as reso
import pipeline.meso as meso
import pipeline.fuse as fuse
import scanreader
import json
def random_transform(input, target, rng, is_rotate=True):
    """
    Randomly rotate/flip the image

    Arguments:
        input: input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: targer image stack (Pytorch Tensor with dimension [b, T, X, Y]), can be None
        rng: numpy random number generator
    
    Returns:
        input: randomly rotated/flipped input image stack (Pytorch Tensor with dimension [b, T, X, Y])
        target: randomly rotated/flipped target image stack (Pytorch Tensor with dimension [b, T, X, Y])
    """
    rand_num = rng.integers(0, 4) # random number for rotation
    rand_num_2 = rng.integers(0, 2) # random number for flip

    if is_rotate:
        if rand_num == 1:
            input = torch.rot90(input, k=1, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=1, dims=(2, 3))
        elif rand_num == 2:
            input = torch.rot90(input, k=2, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=2, dims=(2, 3))
        elif rand_num == 3:
            input = torch.rot90(input, k=3, dims=(2, 3))
            if target is not None:
                target = torch.rot90(target, k=3, dims=(2, 3))
    
    if rand_num_2 == 1:
        input = torch.flip(input, dims=[2])
        if target is not None:
            target = torch.flip(target, dims=[2])

    return input, target


def normalize(image):
    """
    Normalize the image to [mean/std]=[0/1]

    Arguments:
        image: image stack (Pytorch Tensor with dimension [T, X, Y])

    Returns:
        image: normalized image stack (Pytorch Tensor with dimension [T, X, Y])
        mean_image: mean of the image stack (np.float)
        std_image: standard deviation of the image stack (np.float)
    """
    mean_image = torch.mean(image)
    std_image = torch.std(image)

    image -= mean_image
    image /= std_image

    return image, mean_image, std_image


class DatasetSUPPORT(Dataset):
    def __init__(self, noisy_images, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=True, random_patch_seed=0):
        """
        Arguments:
            noisy_images: list of noisy image stack ([Tensor with dimension [t, x, y]])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
            algorithm: the algorithm of use (str)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")      

        # initialize
        self.data_weight = []
        for noisy_image in noisy_images:
            if load_to_memory:
                self.data_weight.append(torch.numel(noisy_image))
            else:
                self.data_weight.append(np.prod(noisy_image.shape))

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.precomputed_indices = None
        self.load_to_memory = load_to_memory

        self.noisy_images = noisy_images
        self.mean_images = []
        self.std_images = []
        if load_to_memory:
            for idx, noisy_image in enumerate(tqdm(noisy_images)):
                noisy_image, mean_image, std_image = normalize(noisy_image)
                self.noisy_images[idx] = noisy_image
                self.mean_images.append(mean_image)
                self.std_images.append(std_image)
            self.mean_images = torch.tensor(self.mean_images)
            self.std_images = torch.tensor(self.std_images)
            print("Normalized noisy images and stored mean/std in memory.")
            print(f"Mean: {self.mean_images}, Std: {self.std_images}")

        # generate index
        self.indices_ds = []
        for noisy_image in self.noisy_images:
            indices = []
            tmp_size = noisy_image.shape
            if np.any(tmp_size < np.array(self.patch_size)):
                raise Exception("patch size is larger than data size")

            for k in range(3):
                z_range = list(range(0, tmp_size[k]-self.patch_size[k]+1, self.patch_interval[k]))
                if tmp_size[k] - self.patch_size[k] > z_range[-1]:
                    z_range.append(tmp_size[k]-self.patch_size[k])
                indices.append(z_range)
            self.indices_ds.append(indices)

    def precompute_indices(self):
        """
        Precompute random patch indices for each image using vectorized operations.
        This function accounts for images of different sizes by generating random
        indices within the valid range for each image.
        """
        precomputed_indices = []
        
        # Iterate over each image in the dataset
        for ds_idx, noisy_image in enumerate(self.noisy_images):
            # Get the shape of the image (T, H, W)
            shape = noisy_image.shape
            
            # Determine the number of patches available for this image.
            # Here, we use the precomputed indices list from __init__ (indices_ds)
            # which was generated based on patch_size and patch_interval.
            indices_lists = self.indices_ds[ds_idx]
            count_i = len(indices_lists[0]) * len(indices_lists[1]) * len(indices_lists[2])
            
            # Calculate the valid range for each dimension
            t_range = shape[0] - self.patch_size[0] + 1
            y_range = shape[1] - self.patch_size[1] + 1
            z_range = shape[2] - self.patch_size[2] + 1
            
            # Generate random indices in a vectorized way for the current image
            t_indices = self.patch_rng.integers(0, t_range, size=count_i)
            y_indices = self.patch_rng.integers(0, y_range, size=count_i)
            z_indices = self.patch_rng.integers(0, z_range, size=count_i)
            
            # Create a list of tuples (ds_idx, t_idx, y_idx, z_idx) for this image
            indices_for_image = [(ds_idx, int(t), int(y), int(z))
                                for t, y, z in zip(t_indices, y_indices, z_indices)]
            precomputed_indices.extend(indices_for_image)
        
        # Shuffle the complete list of precomputed indices to randomize order per epoch
        self.patch_rng.shuffle(precomputed_indices)
        self.precomputed_indices = precomputed_indices

    def __len__(self):
        total = 0
        for indices in self.indices_ds:
            total += len(indices[0]) * len(indices[1]) * len(indices[2])

        return total

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            ds_idx, t_idx, y_idx, z_idx = self.precomputed_indices[i]
        else:
            ds_idx = 0
            t_idx = self.indices_ds[ds_idx][0][i // (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))]
            y_idx = self.indices_ds[ds_idx][1][(i % (len(self.indices_ds[ds_idx][1]) * len(self.indices_ds[ds_idx][2]))) // len(self.indices_ds[ds_idx][2])]
            z_idx = self.indices_ds[ds_idx][2][i % len(self.indices_ds[ds_idx][2])]

        # input dataset range
        t_range = slice(t_idx, t_idx + self.patch_size[0])
        y_range = slice(y_idx, y_idx + self.patch_size[1])
        z_range = slice(z_idx, z_idx + self.patch_size[2])
        
        if self.load_to_memory:
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
        else:
            noisy_image_avg = torch.tensor(self.noisy_images[ds_idx].attrs["mean"])
            noisy_image_std = torch.tensor(self.noisy_images[ds_idx].attrs["std"])
            noisy_image = self.noisy_images[ds_idx][t_range, y_range, z_range]
            noisy_image = torch.tensor(noisy_image, dtype=torch.float32)
            return noisy_image, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
                [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx), noisy_image_avg, noisy_image_std
        
        return noisy_image, torch.tensor([[t_idx, t_idx + self.patch_size[0]],\
            [y_idx, y_idx + self.patch_size[1]], [z_idx, z_idx + self.patch_size[2]]]), torch.tensor(ds_idx)


class DatasetSUPPORT_test_stitch(Dataset):
    def __init__(self, noisy_image, patch_size=[61, 128, 128], patch_interval=[10, 64, 64], load_to_memory=True,\
        transform=None, random_patch=False, random_patch_seed=0):
        """
        Arguments:
            noisy_image: noisy image stack (Tensor with dimension [t, x, y])
            patch_size: size of the patch ([int]), ([t, x, y])
            patch_interval: interval between each patch ([int]), ([t, x, y])
            load_to_memory: whether load data into memory or not (bool)
            transform: function of transformation (function)
            random_patch: sample patch in random or not (bool)
            random_patch_seed: seed for randomness (int)
        """
        # check arguments
        if len(patch_size) != 3:
            raise Exception("length of patch_size must be 3")
        if len(patch_interval) != 3:
            raise Exception("length of patch_interval must be 3")

        self.patch_size = patch_size
        self.patch_interval = patch_interval
        self.transform = transform
        self.random_patch = random_patch
        self.patch_rng = np.random.default_rng(random_patch_seed)
        self.noisy_image = noisy_image
        self.noisy_image, self.mean_image, self.std_image = normalize(self.noisy_image)

        # generate index
        self.indices = []
        tmp_size = self.noisy_image.size()
        if np.any(tmp_size < np.array(self.patch_size)):
            raise Exception("patch size is larger than data size")

        self.indices = get_coordinate(tmp_size, patch_size, patch_interval)            

    def __len__(self):
        return len(self.indices) # len(self.indices[0]) * len(self.indices[1]) * len(self.indices[2])

    def __getitem__(self, i):
        # slicing
        if self.random_patch:
            idx = self.patch_rng.integers(0, len(self.indices) - 1)
        else:
            idx = i
        single_coordinate = self.indices[idx]
        
        # input dataset range
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']

        # for stitching dataset range
        noisy_image = self.noisy_image[init_s:end_s,init_h:end_h,init_w:end_w]

        # transform
        if self.transform:
            rand_i = self.patch_rng.integers(0, self.transform.n_masks)
            rand_t = self.patch_rng.integers(0, 2)
            noisy_image = self.transform.mask(noisy_image, rand_i, rand_t)

        return noisy_image, torch.empty(1), single_coordinate


def gen_train_dataloader(patch_size, patch_interval, batch_size, noisy_data_list, opt, is_zarr=False):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data
    
    Returns:
        dataloader_train
    """
    noisy_images_train = []

    for noisy_data in noisy_data_list:
        if not is_zarr:
            noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(torch.FloatTensor)
            print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
            if len(noisy_image.shape) == 2:
                noisy_image = noisy_image.unsqueeze(0)
            T, _, _ = noisy_image.shape
            noisy_images_train.append(noisy_image)
        else:
            noisy_image = zarr.open(noisy_data, mode='r')
            print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
            noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size,\
        patch_interval=patch_interval, transform=None, random_patch=True, load_to_memory=not is_zarr)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True, prefetch_factor=opt.prefetch_factor)
    
    return dataloader_train


def gen_train_dataloader_nnfabrik(patch_size, patch_interval, batch_size, noisy_data_list, is_zarr=False):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_list: opt.noisy_data

    Returns:
        dataloader_train
    """
    noisy_images_train = []

    for noisy_data in noisy_data_list:
        if not is_zarr:
            noisy_image = torch.from_numpy(skio.imread(noisy_data).astype(np.float32)).type(torch.FloatTensor)
            print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
            if len(noisy_image.shape) == 2:
                noisy_image = noisy_image.unsqueeze(0)
            T, _, _ = noisy_image.shape
            noisy_images_train.append(noisy_image)
        else:
            noisy_image = zarr.open(noisy_data, mode='r')
            print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
            noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size,
                                   patch_interval=patch_interval,
                                   transform=None,
                                   random_patch=True, load_to_memory=not is_zarr)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                  prefetch_factor=2)

    return dataloader_train

def gen_train_dataloader_pipeline(patch_size, patch_interval, batch_size, noisy_data_keys, is_zarr=False):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_keys: opt.noisy_data

    Returns:
        dataloader_train
    """
    noisy_images_train = []

    for key in noisy_data_keys:
        #if not is_zarr:
        noisy_scan = Scan & key
        noisy_data = noisy_scan.local_filenames_as_wildcard
        noisy_image = scanreader.read_scan(noisy_data)
        fuse_mc_keys = (fuse.MotionCorrection & key).fetch(as_dict=True)
        noisy_image_medianidx = int(noisy_image.num_frames // 2) - 1
        median_slice_indices = np.arange(noisy_image_medianidx - 500, noisy_image_medianidx + 500, 1)
        for field_key in fuse_mc_keys:
            which_pipeline = (fuse.MotionCorrection & {}).mapping[field_key['pipe']]


            channel = (reso.CorrectionChannel & field_key).fetch1('channel') if field_key['pipe']=='reso' else \
                (meso.CorrectionChannel & field_key).fetch1('channel')

            raster_correction_params = (reso.RasterCorrection & field_key).fetch1() if field_key['pipe']=='reso' else \
                (meso.RasterCorrection & field_key).fetch1()
            fill_fraction = (reso.ScanInfo & field_key).fetch1('fill_fraction') if field_key['pipe']=='reso' else \
                (meso.ScanInfo & field_key).fetch1('fill_fraction')
            motion_correction_params = (which_pipeline[0] & field_key).fetch1()
            field = motion_correction_params['field']


            noisy_image_field = noisy_image[field-1,:,:,channel-1,median_slice_indices].astype(np.float32)
            if raster_correction_params['raster_phase']>1e-7:
                noisy_image_field = galvo_corrections.correct_raster(noisy_image_field,raster_phase=raster_correction_params['raster_phase'],
                                                                           temporal_fill_fraction=fill_fraction)

            median_slice_xshifts = motion_correction_params['x_shifts'][median_slice_indices]
            median_slice_yshifts = motion_correction_params['y_shifts'][median_slice_indices]
            noisy_image_field = galvo_corrections.correct_motion(noisy_image_field,median_slice_xshifts,median_slice_yshifts)
            noisy_image_field = noisy_image_field.transpose(2, 0, 1)
            noisy_image_field = torch.from_numpy(noisy_image_field).type(torch.FloatTensor)
            print(f"Loaded {noisy_data}")
            if len(noisy_image_field.shape) == 2:
                noisy_image_field = noisy_image_field.unsqueeze(0)

            #T, _, _ = noisy_image_field.shape
            noisy_images_train.append(noisy_image_field)
        # else:
        #     noisy_image = zarr.open(noisy_data, mode='r')
        #     print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
        #     noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size, \
                                   patch_interval=patch_interval,
                                   transform=None,
                                   random_patch=True, load_to_memory=not is_zarr)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, prefetch_factor=2)

    return dataloader_train



def gen_train_dataloader_nnfabrik_pipeline_local(patch_dim, patch_interval_dim, batch_size, noisy_datapath, image_window_length, is_zarr=False):
    """
    Generate dataloader for training

    Arguments:
        patch_size: opt.patch_size
        patch_interval: opt.patch_interval
        noisy_data_keys: opt.noisy_data

    Returns:
        dataloader_train
    """
    patch_size = [61,patch_dim,patch_dim]
    patch_interval = [1,patch_interval_dim,patch_interval_dim]
    with open(noisy_datapath,'r') as f:
        noisy_data_keys = json.load(f)
    noisy_images_train = []

    for key in tqdm(noisy_data_keys):
        #if not is_zarr:
        noisy_scan = Scan & key
        noisy_data = noisy_scan.local_filenames_as_wildcard
        noisy_image = scanreader.read_scan(noisy_data)
        channel = (reso.CorrectionChannel & key).fetch1('channel')

        raster_correction_params = (reso.RasterCorrection & key).fetch1()
        fill_fraction = (reso.ScanInfo & key).fetch1('fill_fraction')
        motion_correction_params = (reso.MotionCorrection & key).fetch1()
        field = motion_correction_params['field']

        noisy_image_medianidx = int(noisy_image.shape[-1] // 2) - 1
        median_slice_indices = np.arange(noisy_image_medianidx - image_window_length, noisy_image_medianidx + image_window_length, 1)
        noisy_image = noisy_image[field-1,:,:,channel-1,median_slice_indices].astype(np.float32)
        if raster_correction_params['raster_phase']>1e-7:
            noisy_image = galvo_corrections.correct_raster(noisy_image,raster_phase=raster_correction_params['raster_phase'],
                                                                       temporal_fill_fraction=fill_fraction)

        median_slice_xshifts = motion_correction_params['x_shifts'][median_slice_indices]
        median_slice_yshifts = motion_correction_params['y_shifts'][median_slice_indices]
        noisy_image = galvo_corrections.correct_motion(noisy_image,median_slice_xshifts,median_slice_yshifts)
        noisy_image = noisy_image.transpose(2, 0, 1)
        noisy_image = torch.from_numpy(noisy_image).type(torch.FloatTensor)
        print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
        if len(noisy_image.shape) == 2:
            noisy_image = noisy_image.unsqueeze(0)

        T, _, _ = noisy_image.shape
        noisy_images_train.append(noisy_image)
        # else:
        #     noisy_image = zarr.open(noisy_data, mode='r')
        #     print(f"Loaded {noisy_data} Shape : {noisy_image.shape}")
        #     noisy_images_train.append(noisy_image)

    dataset_train = DatasetSUPPORT(noisy_images_train, patch_size=patch_size, \
                                   patch_interval=patch_interval,
                                   transform=None,
                                   random_patch=True, load_to_memory=not is_zarr)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, prefetch_factor=2)

    return dataloader_train

def get_nnfabrik_training_dataset(seed: int, **config):
    print(config)
    np.random.seed(seed)
    patch_dim = config.get("patch_dim", 128)
    patch_interval_dim = config.get("patch_interval_dim", 64)
    batch_size = config.get("batch_size", 64)
    dataset_paths = config.get("datasets")
    is_zarr = config.get("is_zarr", False)
    window_length = config.get("window_length",500)
    return {
        "train" : gen_train_dataloader_nnfabrik_pipeline_local(patch_dim,patch_interval_dim,batch_size,dataset_paths,window_length, is_zarr)
    }




