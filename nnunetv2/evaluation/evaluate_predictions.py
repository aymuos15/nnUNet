import multiprocessing
import os
from copy import deepcopy
# from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import cc3d
import torch

import pandas as pd

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# the Evaluator class of the previous nnU-Net was great and all but man was it overengineered. Keep it simple
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

#########################################################################################################
import cupy as cp
from cucim.skimage import measure as cucim_measure

#! GPU connected components
def get_connected_components(img, connectivity=None):

    img_cupy = cp.asarray(img)
    labeled_img, _ = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)

    return labeled_img_torch

#! Dice
# Define the optimized function
def torch_dice(im1, im2):
    
    # Ensure the tensors are on the same device
    im1 = im1.to(im2.device)
    im2 = im2.to(im1.device)

    # Compute Dice coefficient using optimized operations
    intersection = torch.sum(im1 * im2)
    im1_sum = torch.sum(im1)
    im2_sum = torch.sum(im2)
    
    dice_score = (2. * intersection) / (im1_sum + im2_sum)

    return dice_score
#########################################################################################################

def label_or_region_to_key(label_or_region: Union[int, Tuple[int]]):
    return str(label_or_region)


def key_to_label_or_region(key: str):
    try:
        return int(key)
    except ValueError:
        key = key.replace('(', '')
        key = key.replace(')', '')
        split = key.split(',')
        return tuple([int(i) for i in split if len(i) > 0])


def save_summary_json(results: dict, output_file: str):
    """
    json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {label_or_region_to_key(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {label_or_region_to_key(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def load_summary_json(filename: str):
    results = load_json(filename)
    # convert keys in mean metrics
    results['mean'] = {key_to_label_or_region(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results["metric_per_case"])):
        results["metric_per_case"][i]['metrics'] = \
            {key_to_label_or_region(k): results["metric_per_case"][i]['metrics'][k]
             for k in results["metric_per_case"][i]['metrics'].keys()}
    return results


def labels_to_list_of_regions(labels: List[int]):
    return [(i,) for i in labels]


def region_or_label_to_mask(segmentation: np.ndarray, region_or_label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn

def instance_dice(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask_ref = torch.tensor(mask_ref, dtype=torch.long).to(device)
    mask_pred = torch.tensor(mask_pred, dtype=torch.long).to(device)

    for batch_idx in range(mask_ref.shape[0]):
        lbl = mask_ref[batch_idx].cpu().numpy()
        components = cc3d.connected_components(lbl, connectivity=26)
        components = components.astype(np.int32)
        mask_ref[batch_idx] = torch.tensor(components, dtype=torch.long).to(device)
    
    for batch_idx in range(mask_pred.shape[0]):
        pred = mask_pred[batch_idx].cpu().numpy()
        components = cc3d.connected_components(pred, connectivity=26)
        components = components.astype(np.int32)
        mask_pred[batch_idx] = torch.tensor(components, dtype=torch.long).to(device)

    for batch_idx in range(mask_ref.shape[0]):
        pred_cc_volume = mask_pred[batch_idx]
        gt_cc_volume = mask_ref[batch_idx]

        num_lesions = torch.unique(gt_cc_volume[gt_cc_volume != 0]).size(0)

        lesion_dice_scores = torch.tensor([0.0]).to(device)
        tp = torch.tensor([]).to(device)

        for gtcomp in range(1, num_lesions + 1):
            gt_tmp = (gt_cc_volume == gtcomp)
            intersecting_cc = torch.unique(pred_cc_volume[gt_tmp])
            intersecting_cc = intersecting_cc[intersecting_cc != 0]

            if len(intersecting_cc) > 0:
                pred_tmp = torch.zeros_like(pred_cc_volume, dtype=torch.bool)
                pred_tmp[torch.isin(pred_cc_volume, intersecting_cc)] = True
                dice_score = torch_dice(pred_tmp, gt_tmp)
                lesion_dice_scores += dice_score
                tp = torch.cat([tp, intersecting_cc])
            else:
                lesion_dice_scores += torch.tensor([0.0]).to(device)
        
        mask = (pred_cc_volume != 0) & (~torch.isin(pred_cc_volume, tp))
        fp = torch.unique(pred_cc_volume[mask], sorted=True).to(device)
        fp = fp[fp != 0]

        if num_lesions + len(fp) > 0:
            volume_dice_score = lesion_dice_scores / (num_lesions + len(fp))
        else:
            volume_dice_score = torch.tensor([0.0])

        count = torch.tensor([num_lesions - len(tp)])

        volume_dice_score = volume_dice_score.cpu().numpy()
        count = count.cpu().numpy()

    return volume_dice_score, count

# import pandas as pd
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def dice_torch(im1, im2):
#     intersection = torch.sum(im1 * im2)
#     sum_im1 = torch.sum(im1)
#     sum_im2 = torch.sum(im2)
#     return 2.0 * intersection / (sum_im1 + sum_im2)

# def collect_legacy_metrics(pred_label_cc, gt_label_cc):
#     legacy_metrics = []
#     tp = torch.tensor([], device=device)
#     fn = torch.tensor([], device=device)

#     num_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)

#     for gtcomp in range(1, num_gt_lesions + 1):
#         gt_tmp = (gt_label_cc == gtcomp)
#         intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
#         intersecting_cc = intersecting_cc[intersecting_cc != 0]

#         for cc in intersecting_cc:
#             tp = torch.cat([tp, torch.tensor([cc], device=device)])
#             legacy_metrics.append({'GT': gtcomp, 'Pred': cc.item(), 'Dice': dice_torch(pred_label_cc == cc, gt_tmp)})

#         if len(intersecting_cc) == 0:
#             legacy_metrics.append({'GT': gtcomp, 'Pred': 0, 'Dice': 0})
#             fn = torch.cat([fn, torch.tensor([gtcomp], device=device)])

#     zero_tensor = torch.tensor([0], device=device)
#     fp = torch.unique(pred_label_cc[torch.isin(pred_label_cc, torch.cat((tp, zero_tensor)), invert=True)])
#     fp = fp[fp != 0]
#     return legacy_metrics, tp, fp, fn

# def find_overlapping_components(prediction_cc, gt_cc):
#     overlapping_components = {}
#     overlapping_components_inverse = {}

#     # Iterate over all non-zero elements in the prediction_cc tensor
#     for i, j, k in zip(*torch.nonzero(prediction_cc, as_tuple=True)):
#         prediction_component = prediction_cc[i, j, k].item()
#         gt_component = gt_cc[i, j, k].item()

#         if prediction_component != 0 and gt_component != 0:
#             if prediction_component not in overlapping_components:
#                 overlapping_components[prediction_component] = set()
#             overlapping_components[prediction_component].add(gt_component)

#             if gt_component not in overlapping_components_inverse:
#                 overlapping_components_inverse[gt_component] = set()
#             overlapping_components_inverse[gt_component].add(prediction_component)

#     # Filter out entries with only one overlapping component
#     overlapping_components = {k: v for k, v in overlapping_components.items() if len(v) > 1}
#     overlapping_components_inverse = {k: v for k, v in overlapping_components_inverse.items() if len(v) > 1}

#     return overlapping_components, overlapping_components_inverse

# def generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components):
#     overlap_metrics = []

#     for pred_component, gt_components in overlapping_components.items():
#         gtcomps = list(gt_components)
#         pred_cc_tmp = (pred_label_cc == pred_component).to(torch.int32)
#         gt_cc_tmp = (gt_label_cc[..., None] == torch.tensor(gtcomps, device=gt_label_cc.device)).any(-1).to(torch.int32)
#         overlap_metrics.append({'GT': gtcomps, 'Pred': pred_component, 'Dice': dice_torch(pred_cc_tmp, gt_cc_tmp)})

#     return overlap_metrics

# def generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components):
#     overlap_metrics = []

#     for gt_component, pred_components in overlapping_components.items():
#         predcomps = list(pred_components)
#         gt_cc_tmp = (gt_label_cc == gt_component).to(torch.int32)
#         pred_cc_tmp = (pred_label_cc[..., None] == torch.tensor(predcomps, device=pred_label_cc.device)).any(-1).to(torch.int32)
#         overlap_metrics.append({'GT': gt_component, 'Pred': predcomps, 'Dice': dice_torch(pred_cc_tmp, gt_cc_tmp)})

#     return overlap_metrics

# def collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse):
#     legacy_metrics, tp, fp, fn = collect_legacy_metrics(pred_label_cc, gt_label_cc)
#     legacy_metrics = pd.DataFrame(legacy_metrics)
    
#     overlap_metrics = generate_overlap_metrics(pred_label_cc, gt_label_cc, overlapping_components)
#     overlap_metrics = pd.DataFrame(overlap_metrics)
    
#     overlap_metrics_inverse = generate_overlap_metrics_inverse(pred_label_cc, gt_label_cc, overlapping_components_inverse)
#     overlap_metrics_inverse = pd.DataFrame(overlap_metrics_inverse)

    
#     initial_metrics_df = pd.concat([legacy_metrics, overlap_metrics, overlap_metrics_inverse], ignore_index=True)
#     return initial_metrics_df, tp, fp, fn

# def process_metric_df(df):
#     if df.empty:
#         return []
#     else:
#         gt_list = []
#         pred_list = []
#         for gt, pred in zip(df['GT'], df['Pred']):
#             if isinstance(gt, list):
#                 gt_list.extend(gt)
#             if isinstance(pred, list):
#                 pred_list.extend(pred)
#         combined = set(gt_list + pred_list)
#         indices_to_drop = []
#         # if statement to chheck if ground truth is even present in the combined set
#         for idx, (gt, pred) in enumerate(zip(df['GT'], df['Pred'])):
#             if isinstance(gt, int) and gt in combined and isinstance(pred, int):
#                 indices_to_drop.append(idx)
#         df.drop(indices_to_drop, inplace=True)
#         df['GT'] = df['GT'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
#         df['Pred'] = df['Pred'].apply(lambda x: tuple(x) if isinstance(x, list) else x)
#         df.drop_duplicates(subset=['GT', 'Pred'], inplace=True)
#         return df['Dice'].to_list()

# def instance_dice(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     mask_ref = torch.tensor(mask_ref, dtype=torch.long).to(device)
#     mask_pred = torch.tensor(mask_pred, dtype=torch.long).to(device)

#     for batch_idx in range(mask_ref.shape[0]):
#         lbl = mask_ref[batch_idx].cpu().numpy()
#         components = cc3d.connected_components(lbl, connectivity=26)
#         components = components.astype(np.int32)
#         mask_ref[batch_idx] = torch.tensor(components, dtype=torch.long).to(device)
    
#     for batch_idx in range(mask_pred.shape[0]):
#         pred = mask_pred[batch_idx].cpu().numpy()
#         components = cc3d.connected_components(pred, connectivity=26)
#         components = components.astype(np.int32)
#         mask_pred[batch_idx] = torch.tensor(components, dtype=torch.long).to(device)

#     total_dice_scores = torch.tensor([]).to(device)

#     for batch in range(mask_ref.shape[0]):
#         pred_label_cc = mask_pred[batch]
#         gt_label_cc = mask_ref[batch]

#         overlapping_components, overlapping_components_inverse = find_overlapping_components(pred_label_cc, gt_label_cc)    
#         final_metric, tp, fp, fn = collect_all_metrics(pred_label_cc, gt_label_cc, overlapping_components, overlapping_components_inverse)
#         dice_score = process_metric_df(final_metric)

#         if len(dice_score) == 0:
#             dice_score = torch.tensor([0.0]).to(device)
#         else:
#             final_score = sum(dice_score) / (len(dice_score) + len(fp))
#             total_dice_scores = torch.cat([total_dice_scores, torch.tensor([final_score]).to(device)])
    
#     final_dice_score = torch.mean(total_dice_scores) 
        
#     return final_dice_score, torch.tensor([0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cluster_dice(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):

    mask_ref = torch.tensor(mask_ref, dtype=torch.long).to(device)
    mask_pred = torch.tensor(mask_pred, dtype=torch.long).to(device)

    overlay = mask_pred + mask_ref
    overlay[overlay > 0] = 1

    labelled_overlay = torch.zeros_like(overlay)

    overlay = overlay.to(mask_pred.device)
    labelled_overlay = labelled_overlay.to(mask_pred.device)

    overlay = overlay.detach().cpu().numpy()
    for batch in range(mask_pred.shape[0]):
            overlay_batch = overlay[batch]
            labelled_overlay_batch = cc3d.connected_components(overlay_batch, connectivity=26)
            labelled_overlay_batch = labelled_overlay_batch.astype(np.int32)
            labelled_overlay[batch] = torch.tensor(labelled_overlay_batch, dtype=torch.long).to(mask_pred.device)

    num_clusters = torch.unique(labelled_overlay[labelled_overlay != 0]).size(0)

    score_tally = torch.tensor([]).to(mask_pred.device)

    for cluster in range(1, num_clusters + 1):
        cluster_mask = (labelled_overlay == cluster).float()

        pred_cluster = torch.logical_and(mask_pred, cluster_mask).to(mask_pred.device)
        gt_cluster = torch.logical_and(mask_ref, cluster_mask).to(mask_pred.device)

        score = torch_dice(pred_cluster, gt_cluster)
        score_tally = torch.cat((score_tally, score.unsqueeze(0)))
    
    return torch.mean(score_tally), torch.tensor([0])

def compute_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter,
                    labels_or_regions: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                    ignore_label: int = None) -> dict:
    # load images
    seg_ref, seg_ref_dict = image_reader_writer.read_seg(reference_file)
    seg_pred, seg_pred_dict = image_reader_writer.read_seg(prediction_file)
    # spacing = seg_ref_dict['spacing']

    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        lesion_dice, _ = cluster_dice(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
        results['metrics'][r]['Lesion_Dice'] = float(lesion_dice)
    return results


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              image_reader_writer: BaseReaderWriter,
                              file_ending: str,
                              regions_or_labels: Union[List[int], List[Union[int, Tuple[int, ...]]]],
                              ignore_label: int = None,
                              num_processes: int = default_num_processes,
                              chill: bool = True) -> dict:
    """
    output_file must end with .json; can be None
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        # for i in list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred), [ignore_label] * len(files_pred))):
        #     compute_metrics(*i)
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [image_reader_writer] * len(files_pred), [regions_or_labels] * len(files_pred),
                     [ignore_label] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    [recursive_fix_for_json_export(i) for i in results]
    recursive_fix_for_json_export(means)
    recursive_fix_for_json_export(foreground_mean)
    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    return result
    # print('DONE')


def compute_metrics_on_folder2(folder_ref: str, folder_pred: str, dataset_json_file: str, plans_file: str,
                               output_file: str = None,
                               num_processes: int = default_num_processes,
                               chill: bool = False):
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')

    lm = PlansManager(plans_file).get_label_manager(dataset_json)
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              lm.foreground_regions if lm.has_regions else lm.foreground_labels, lm.ignore_label,
                              num_processes, chill=chill)


def compute_metrics_on_folder_simple(folder_ref: str, folder_pred: str, labels: Union[Tuple[int, ...], List[int]],
                                     output_file: str = None,
                                     num_processes: int = default_num_processes,
                                     ignore_label: int = None,
                                     chill: bool = False):
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'summary.json')
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                              labels, ignore_label=ignore_label, num_processes=num_processes, chill=chill)


def evaluate_folder_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_metrics_on_folder2(args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill)


def evaluate_simple_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt segmentations')
    parser.add_argument('pred_folder', type=str, help='folder with predicted segmentations')
    parser.add_argument('-l', type=int, nargs='+', required=True,
                        help='list of labels')
    parser.add_argument('-il', type=int, required=False, default=None,
                        help='ignore label')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.l, args.o, args.np, args.il, chill=args.chill)


if __name__ == '__main__':
    folder_ref = '/media/fabian/data/nnUNet_raw/Dataset004_Hippocampus/labelsTr'
    folder_pred = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation'
    output_file = '/home/fabian/results/nnUNet_remake/Dataset004_Hippocampus/nnUNetModule__nnUNetPlans__3d_fullres/fold_0/validation/summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    regions = labels_to_list_of_regions([1, 2])
    ignore_label = None
    num_processes = 12
    compute_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, file_ending, regions, ignore_label,
                              num_processes)
