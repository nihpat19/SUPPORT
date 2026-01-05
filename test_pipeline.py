import numpy as np
import scanreader
import torch
import skimage.io as skio

from tqdm import tqdm
from src.utils.dataset import DatasetSUPPORT_test_stitch
from model.SUPPORT import SUPPORT
import pipeline.experiment as experiment
import pipeline.reso as reso
import pipeline.meso as meso
import pipeline.fuse as fuse
import json
from pipeline.utils import galvo_corrections

def validate(test_dataloader, model):
    """
    Validate a model with a test data
    
    Arguments:
        test_dataloader: (Pytorch DataLoader)
            Should be DatasetFRECTAL_test_stitch!
        model: (Pytorch nn.Module)

    Returns:
        denoised_stack: denoised image stack (Numpy array with dimension [T, X, Y])
    """
    with torch.no_grad():
        model.eval()
        # initialize denoised stack to NaN array.
        denoised_stack = np.zeros(test_dataloader.dataset.noisy_image.shape, dtype=np.float32)
        
        # stitching denoised stack
        # insert the results if the stack value was NaN
        # or, half of the output volume
        for _, (noisy_image, _, single_coordinate) in enumerate(tqdm(test_dataloader, desc="validate")):
            noisy_image = noisy_image.cuda() #[b, z, y, x]
            noisy_image_denoised = model(noisy_image)
            T = noisy_image.size(1)
            for bi in range(noisy_image.size(0)): 
                stack_start_w = int(single_coordinate['stack_start_w'][bi])
                stack_end_w = int(single_coordinate['stack_end_w'][bi])
                patch_start_w = int(single_coordinate['patch_start_w'][bi])
                patch_end_w = int(single_coordinate['patch_end_w'][bi])

                stack_start_h = int(single_coordinate['stack_start_h'][bi])
                stack_end_h = int(single_coordinate['stack_end_h'][bi])
                patch_start_h = int(single_coordinate['patch_start_h'][bi])
                patch_end_h = int(single_coordinate['patch_end_h'][bi])

                stack_start_s = int(single_coordinate['init_s'][bi])
                
                denoised_stack[stack_start_s+(T//2), stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                    = noisy_image_denoised[bi].squeeze()[patch_start_h:patch_end_h, patch_start_w:patch_end_w].cpu()

        # change nan values to 0 and denormalize
        denoised_stack = denoised_stack * test_dataloader.dataset.std_image.numpy() + test_dataloader.dataset.mean_image.numpy()

        return denoised_stack


if __name__ == '__main__':
    ########## Change it with your data ##############
    data_file = "../single_cell_key2.json"
    model_file = "../training_results/saved_models/pipeline_dataloading_testfull/model_499.pth"
    output_file = "../testing_results/single_cell_notpartoftestingpipeline_differentmodel"
    patch_size = [61, 16, 16]
    patch_interval = [1, 8, 8]
    batch_size = 3200    # lower it if memory exceeds.
    bs_size = 3    # modify if you changed bs_size when training.
    bp_mode = False
    ##################################################

    model = SUPPORT(in_channels=61, mid_channels=[64, 128, 256, 512, 1024], depth=5,\
            blind_conv_channels=64, one_by_one_channels=[32, 16], last_layer_channels=[64, 32, 16], bs_size=bs_size, bp=bp_mode).cuda()

    model.load_state_dict(torch.load(model_file))
    demo_key = json.load(open(data_file, 'r'))
    noisy_scan = experiment.Scan & demo_key
    noisy_data = noisy_scan.local_filenames_as_wildcard
    #noisy_image = scanreader.read_scan(noisy_data)
    print(f"Testing scan {noisy_data}")
    fuse_mc_keys = (fuse.MotionCorrection & demo_key).fetch(as_dict=True)
    for field_key in fuse_mc_keys:
        which_pipeline = (fuse.MotionCorrection & {}).mapping[field_key['pipe']]

        channel = (reso.CorrectionChannel & field_key).fetch1('channel') if field_key['pipe'] == 'reso' else \
            (meso.CorrectionChannel & field_key).fetch1('channel')

        raster_correction_params = (reso.RasterCorrection & field_key).fetch1() if field_key['pipe'] == 'reso' else \
            (meso.RasterCorrection & field_key).fetch1()
        fill_fraction = (reso.ScanInfo & field_key).fetch1('fill_fraction') if field_key['pipe'] == 'reso' else \
            (meso.ScanInfo & field_key).fetch1('fill_fraction')
        motion_correction_params = (which_pipeline[0] & field_key).fetch1()
        field = motion_correction_params['field']

        noisy_image_field = scanreader.read_scan(noisy_data)[field - 1, :, :, channel - 1, :].astype(np.float32)
        f"Loaded scan field {field}. Performing Raster and motion correction..."
        if raster_correction_params['raster_phase'] > 1e-7:
            noisy_image_field = galvo_corrections.correct_raster(noisy_image_field,
                                                                 raster_phase=raster_correction_params['raster_phase'],
                                                                 temporal_fill_fraction=fill_fraction)

        xshifts = motion_correction_params['x_shifts']
        yshifts = motion_correction_params['y_shifts']
        noisy_image_field = galvo_corrections.correct_motion(noisy_image_field, xshifts,
                                                             yshifts)
        noisy_image_field = noisy_image_field.transpose(2, 0, 1)
        noisy_image_field = torch.from_numpy(noisy_image_field).type(torch.FloatTensor)
        testset = DatasetSUPPORT_test_stitch(noisy_image_field, patch_size=patch_size, \
                                             patch_interval=patch_interval)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
        denoised_stack = validate(testloader, model)
        skio.imsave(f'{output_file}_field{field}.tif', denoised_stack[(model.in_channels - 1) // 2:-(model.in_channels - 1) // 2, :, :],
                    metadata={'axes': 'TYX'})

    # channel = (reso.CorrectionChannel & demo_key).fetch1('channel')
    #
    # raster_correction_params = (reso.RasterCorrection & demo_key).fetch1()
    # fill_fraction = (reso.ScanInfo & demo_key).fetch1('fill_fraction')
    # motion_correction_params = (reso.MotionCorrection & demo_key).fetch1()
    # field = motion_correction_params['field']
    # noisy_image = noisy_image[field - 1, :, :, channel - 1, :].astype(np.float32)
    # print("Performing motion correction...")
    # if raster_correction_params['raster_phase'] > 1e-7:
    #     noisy_image = galvo_corrections.correct_raster(noisy_image,
    #                                                    raster_phase=raster_correction_params['raster_phase'],
    #                                                    temporal_fill_fraction=fill_fraction)
    # noisy_image = galvo_corrections.correct_motion(noisy_image,motion_correction_params['x_shifts'],motion_correction_params['y_shifts'])
    # noisy_image = noisy_image.transpose(2, 0, 1)
    # noisy_image = torch.from_numpy(noisy_image).type(torch.FloatTensor)
    #
    # testset = DatasetSUPPORT_test_stitch(noisy_image, patch_size=patch_size,\
    #     patch_interval=patch_interval)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    # denoised_stack = validate(testloader, model)
    #
    # print(denoised_stack.shape)
    # skio.imsave(output_file, denoised_stack[(model.in_channels-1)//2:-(model.in_channels-1)//2, : , :], metadata={'axes': 'TYX'})
