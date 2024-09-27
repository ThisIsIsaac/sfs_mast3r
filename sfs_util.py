"""If there are errors on importing, try running this first:

import os
os.chdir("/viscam/u/iamisaac/mast3r")
current_path = os.getcwd()
print("Current working directory:", current_path)
import mast3r
import inspect
import importlib
importlib.reload(mast3r)
"""
import os
import sys
import mast3r.utils.path_to_dust3r
sys.path.append('/viscam/u/iamisaac/mast3r/dust3r/croco')
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.demo as demo
import wandb
from dust3r.inference import inference
from dust3r.utils.image import load_images
# visualize a few matches
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np
import numpy as np
import requests
from PIL import Image
import base64
import io
import pandas as pd
import ast
import copy
import mast3r.demo
from clip_retrieval.clip_back import download_image
import time
import imghdr
from urllib.parse import urlparse
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import json
from datetime import datetime
import shutil

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="sfs",
)

def init_mast3r(device='cuda'):
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return model

def run_mast3r(model1, img1_path, img2_path, device='cuda'):
    images = load_images([img1_path, img2_path], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    figure_img = viz_matches(output, matches_im0, matches_im1)
    return (output, matches_im0, matches_im1, figure_img)

def viz_matches(mast3r_output, matches_im0, matches_im1, n_viz = 20):
    view1, view2 = mast3r_output['view1'], mast3r_output['view2']
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

    # Convert the buffer to a PIL Image
    buf.seek(0)
    pil_image = Image.open(buf)
    return pil_image


def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = torch.from_numpy(R)[None, :, :]
    T = torch.from_numpy(T)[None, :]
    focal = torch.from_numpy(focal)[None, :]
    p0 = torch.from_numpy(p0)[None, :]
    image_size = torch.from_numpy(image_size)[None, :]

    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.permute(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # NDC to screen conversion.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = torch.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]


def rotation_error(R_out, R_gt):
    trace = np.trace(np.dot(R_out.T, R_gt))
    error = np.arccos((trace - 1) / 2)

    return error


def compute_camera_transformation(R1, T1, R2, T2):
    """
    Compute the transformation between two cameras given their rotations and translations.

    :param R1: 3x3 rotation matrix for the first camera
    :param T1: 3x1 translation vector for the first camera
    :param R2: 3x3 rotation matrix for the second camera
    :param T2: 3x1 translation vector for the second camera
    :return: tuple (transform, R, T)
             transform: 4x4 transformation matrix from camera 1 to camera 2
             R: 3x3 rotation matrix component of the transformation
             T: 3x1 translation vector component of the transformation
    """
    # Ensure inputs are numpy arrays
    R1, T1, R2, T2 = map(np.array, [R1, T1, R2, T2])

    # Construct 4x4 camera-to-world matrices
    c2w_1 = np.eye(4)
    c2w_1[:3, :3] = R1
    c2w_1[:3, 3] = T1.flatten()

    c2w_2 = np.eye(4)
    c2w_2[:3, :3] = R2
    c2w_2[:3, 3] = T2.flatten()

    # Compute world-to-camera matrices (inverse of camera-to-world)
    w2c_2 = np.linalg.inv(c2w_2)

    # The transformation from camera 1 to camera 2 is:
    # First transform from camera 1 to world (c2w_1),
    # then from world to camera 2 (w2c_2)
    transform_1_to_2 = w2c_2 @ c2w_1

    # Extract R and T from the transformation matrix
    R = transform_1_to_2[:3, :3]
    T = transform_1_to_2[:3, 3]

    return R, T

def translation_error(T_out, T_gt):
    # Calculate the scale (magnitude of the ground truth translation)
    scene_scale = np.linalg.norm(T_gt)

    frame_error = np.linalg.norm(T_out - T_gt)
    error = frame_error / scene_scale

    return error

def get_errors(pred_Rs, pred_Ts, gt_Rs, gt_Ts):
    R_err = []
    T_err = []
    for i in range(len(pred_Rs)):
        R_err.append(rotation_error(pred_Rs[i], gt_Rs[i]))
        T_err.append(translation_error(pred_Ts[i], gt_Ts[i]))
    pred_transform_R, pred_transform_T = compute_camera_transformation(pred_Rs[0], pred_Ts[0], pred_Rs[1], pred_Ts[1])
    gt_transform_R, gt_transform_T = compute_camera_transformation(gt_Rs[0], gt_Ts[0], gt_Rs[1], gt_Ts[1])
    R_transform_err = rotation_error(pred_transform_R, gt_transform_R)
    T_transform_err = translation_error(pred_transform_T, gt_transform_T)
    return R_transform_err, T_transform_err, R_err, T_err

def is_valid_image_url(url):
    try:
        # Check if the URL is valid
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        # Send a HEAD request to check the content type
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get('content-type', '')

        # Check if the content type is an image
        if 'image' not in content_type:
            return False

        # If content-type check passes, try to get a small part of the image
        response = requests.get(url, stream=True, timeout=5)
        content = next(response.iter_content(chunk_size=1024))
        
        # Use imghdr to check if the content is a valid image
        image_type = imghdr.what(None, content)
        return image_type is not None

    except Exception as e:
        print(f"Error checking URL {url}: {str(e)}")
        return False

def send_query(response_idxs, image_path=None, query_text=None, display_images=False, num_responses = 5, indice_name="co3d", image_cache_path=None, include_query_image=False):
    data = {
        "num_images": num_responses,
        "num_result_ids": num_responses,
        "indice_name": indice_name,
        "modality":"image"
    }
    if query_text:
        data["text"] = query_text
    else:
        with open(image_path, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode('utf-8')
        data["image"] = image
    response = requests.post("http://localhost:1234/knn-service", json=data)
    if response.status_code != 200:
        raise ValueError("Request failed!!")
    results = parse_response(response, response_idxs=response_idxs, indice_name=indice_name, image_cache_path=image_cache_path, include_query_image=include_query_image, query_image_path=image_path)
    return results

def parse_response(response, indice_name, response_idxs, image_cache_path=None, include_query_image=False, query_image_path=None):
    results = response.json()
    results = [results[idx] for idx in response_idxs]
    parsed_results = []
    if include_query_image:
        parsed_results.append({
            "image_path": query_image_path,
            "image": np.array(Image.open(query_image_path))
        })
    for result in results:
        parsed_result = copy.deepcopy(result)
        if "image" in result:
            image = Image.open(io.BytesIO(base64.b64decode(result["image"])))
        elif "url" in result:
            # Generate a unique filename
            filename = f"image_{result.get('id', hash(result['url']))}.jpg"
            filepath = os.path.join(image_cache_path, filename)
            
            if is_valid_image_url(result["url"]):
                try:
                    # Download and save the image
                    image_data = download_image(result["url"])
                    if isinstance(image_data, io.BytesIO):
                        image = Image.open(image_data).resize((256, 256))
                    else:
                        image = Image.open(io.BytesIO(image_data)).resize((256, 256))
                    image.save(filepath)
                    parsed_result["image_path"] = filepath
                except Exception as e:
                    print(f"Error downloading or saving image from url: {result['url']}")
                    print(f"Error details: {str(e)}")
                    continue
            else:
                print(f"Invalid or non-image URL: {result['url']}")
                continue

        parsed_result["image"] = np.array(image)
        if indice_name == "co3d":
            R = ast.literal_eval(result["R"])
            R = np.array(R).reshape(3, 3)
            T = ast.literal_eval(result["T"])
            T = np.array(T)
            focal_length = np.array(ast.literal_eval(result["focal_length"]))
            principal_point = np.array(ast.literal_eval(result["principal_point"]))

            R, T, camera_intrinsics = opencv_from_cameras_projection(R,
                                                                        T,
                                                                        focal_length,
                                                                        principal_point,
                                                                        np.array(image.size))
            parsed_result["R"] = R
            parsed_result["T"] = T
            parsed_result["focal_length"] = focal_length
            parsed_result["principal_point"] = principal_point
        parsed_results.append(parsed_result)
    return parsed_results

def run_reconstruction(model, filelist, min_conf_thr = 1.5, matching_conf_thr=2.0, shared_intrinsics=False, output_path="/viscam/projects/sfs/mast3r_outputs/test1"):
    gradio_delete_cache = False
    current_scene_state = None # gradio scene state
    image_size = 512 #224
    optim_level = "refine"
    silent = True
    device = 'cuda'
    niter1 = 500
    lr1 = 0.07
    niter2 = 200
    lr2 = 0.014
    as_pointcloud = True
    mask_sky = True
    clean_depth = True
    transparent_cams = True
    cam_size = 0.2
    scenegraph_type = "complete"
    winsize = 1
    win_cyclic = False
    TSDF_thresh = 0
    refid = 0 # Scene ID

    args = {
        "outdir":output_path,
        "model":model,
        "device":device,
        "filelist":filelist,
        "niter1":niter1,
        "niter2":niter2,
        "lr1":lr1,
        "lr2":lr2,
        "as_pointcloud":as_pointcloud,
        "cam_size":cam_size,
        "TSDF_thresh":TSDF_thresh,
        "gradio_delete_cache":gradio_delete_cache,
        "current_scene_state":current_scene_state,
        "image_size":image_size,
        "optim_level":optim_level,
        "silent":silent,
        "mask_sky":mask_sky,
        "clean_depth":clean_depth,
        "transparent_cams":transparent_cams,
        "scenegraph_type":scenegraph_type,
        "winsize":winsize,
        "win_cyclic":win_cyclic,
        "refid":refid,
        "shared_intrinsics":shared_intrinsics,
        "min_conf_thr":min_conf_thr,
        "matching_conf_thr":matching_conf_thr,
    }
    sparse_ga_state, outfile = demo.get_reconstructed_scene(**args)
    sparse_ga_state.sparse_ga.recon_file_path = outfile
    return sparse_ga_state.sparse_ga

def run_mast3r_from_clip_retrieval(model, output_path, image_path=None, query_text=None, num_responses=10, response_idxs=[0, 1], min_conf_thr = 1.5, matching_conf_thr=2.0, shared_intrinsics=False, display_images=False, indice_name="co3d", image_cache_path=None, include_query_image=False):
    responses = send_query(image_path=image_path, query_text=query_text, display_images=display_images, num_responses=num_responses, response_idxs=response_idxs, indice_name=indice_name, image_cache_path=image_cache_path, include_query_image=include_query_image)
    img_paths = [response["image_path"] for response in responses]
    images = [wandb.Image(response["image"]) for response in responses]
    if len(images) < 2:
        print(f"Not enough images to run reconstruction!")
        return
    sparse_ga = run_reconstruction(model, img_paths, min_conf_thr=min_conf_thr, matching_conf_thr=matching_conf_thr, shared_intrinsics=shared_intrinsics, output_path=output_path)
    if not sparse_ga.recon_file_path:
        if image_path:
            print(f"Matching failed for {image_path}!")
        else:
            print(f"Matching failed for '{query_text}'!")
        return
    mask1, mask2 = sparse_ga.masks
    dinov2_score = get_dinov2_score(img_paths[0], img_paths[1], sparse_ga.cache_path, mask1, mask2)
    sparse_ga.dinov2_score = dinov2_score
    wandb_log = {
        "mast3r_results": wandb.Object3D(open(sparse_ga.recon_file_path)),
        "input images": images,
        "num_views": len(images),
        "min_conf_thr": min_conf_thr,
        "matching_conf_thr": matching_conf_thr,
        "dinov2_score": dinov2_score,
    }

    if indice_name == "co3d":
        gt_R = [response["R"] for response in responses]
        gt_T = [response["T"] for response in responses]
        pred_R = [cam2w[:3, :3].cpu() for cam2w in sparse_ga.cam2w]
        pred_T = [cam2w[3:, 3:].cpu() for cam2w in sparse_ga.cam2w]
        R_transform_err, T_transform_err, R_err, T_err = get_errors(pred_R, pred_T, gt_R, gt_T)
        wandb_log.update({
            "image paths":str(img_paths),
            "query_image_path":image_path,
            "R_transform_err":R_transform_err,
            "T_transform_err":T_transform_err,
            "R_cam0_err":R_err[0],
            "T_cam0_err":T_err[0],
            "R_cam1_err": R_err[1],
            "T_cam1_err": T_err[1],
        })
    wandb.log(wandb_log)
    return sparse_ga

def download_images_from_datacomp(output_path, query_texts=None, query_img_paths=None, num_responses=10,):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    download_dir = os.path.join(output_path, f"downloaded_images_{timestamp}")
    
    os.makedirs(download_dir, exist_ok=True)
    if query_texts:
        for query_text in query_texts:
            query_dir = query_text.replace(" ", "_")
            query_path = os.path.join(download_dir, query_dir)
            os.makedirs(query_path, exist_ok=True)
            for i in range(0, num_responses-1):
                send_query(None, [i, i+1], query_text=query_text, num_responses = num_responses, indice_name="datacomp", image_cache_path=query_path)
    
    else:
        for img_idx, query_img_path in enumerate(query_img_paths):
            query_path = os.path.join(download_dir, str(img_idx))
            os.makedirs(query_path, exist_ok=True)
            for i in range(0, num_responses-1):
                send_query(image_path=query_img_path, response_idxs=[i, i+1], num_responses = num_responses, indice_name="datacomp", image_cache_path=query_path)


def get_dinov2_score(img1_path, img2_path, cache_path, mask1, mask2, device='cuda'):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dinov2_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    mask1 = torch.tensor(mask1, dtype=torch.bool, device=device)
    mask2 = torch.tensor(mask2, dtype=torch.bool, device=device)
    height, width = mask1.shape

    
    inputs1 = processor(images=img1, return_tensors="pt").to(device)
    out1 = dinov2_model(**inputs1)
    last_hidden1 = out1[0]
    
    inputs2 = processor(images=img2, return_tensors="pt").to(device)
    last_hidden2 = dinov2_model(**inputs2)[0]

    # Remove the [CLS] token
    last_hidden1 = last_hidden1[:, 1:, :]  # Shape: [1, 256, 768]
    last_hidden2 = last_hidden2[:, 1:, :]  # Shape: [1, 256, 768]

    # Reshape to a square grid (assuming 14x14 patches)
    patch_size = 16
    feautre_dim = last_hidden1.shape[-1]
    reshaped_output1 = last_hidden1.reshape(1, patch_size, patch_size, feautre_dim)
    reshaped_output1 = reshaped_output1.permute(0, 3, 1, 2)
    reshaped_output1 = F.interpolate(reshaped_output1, size=(height, width), mode='bilinear')
    reshaped_output2 = last_hidden2.reshape(1, patch_size, patch_size, feautre_dim)
    reshaped_output2 = reshaped_output2.permute(0, 3, 1, 2)
    reshaped_output2 = F.interpolate(reshaped_output2, size=(height, width), mode='bilinear')
    
    """
    [
        [conf_score, sum of point-wise conf, number of mutual nn pairs],
        [xy1, xy2, point-wise conf]
    ]
    """
    xy1, xy2, _ = torch.load(cache_path)[1]
    # Create boolean masks for valid points
    valid_mask1 = mask1[xy1[:, 1], xy1[:, 0]]
    valid_mask2 = mask2[xy2[:, 1], xy2[:, 0]]

    # Combine masks
    valid_mask = valid_mask1 & valid_mask2

    # Apply the mask to get the filtered coordinates
    masked_xy1 = xy1[valid_mask]
    masked_xy2 = xy2[valid_mask]
        
    # Extract features for xy1 from reshaped_output1
    features1 = reshaped_output1[0, :, masked_xy1[:, 1], masked_xy1[:, 0]]   

    # Extract features for xy2 from reshaped_output2
    features2 = reshaped_output2[0, :, masked_xy2[:, 1], masked_xy2[:, 0]]

    similarity = F.cosine_similarity(features1, features2, dim=0)
    similarity_score = similarity.mean()

    return similarity_score.item()


def create_html(img1_paths, img2_paths, model_paths, scores, output_html_dir):
    data_list = []
    current_file_path = os.path.abspath(__file__)
    for idx in range(len(img1_paths)):
        img1_path = img1_paths[idx]
        img2_path = img2_paths[idx]
        
        # Check if img1 and img2 are in the same directory
        if os.path.dirname(img1_path) != os.path.dirname(img2_path):
            # If not, copy img1 to img2's directory
            new_img1_path = os.path.join(os.path.dirname(img2_path), os.path.basename(img1_path))
            shutil.copy2(img1_path, new_img1_path)
            img1_path = new_img1_path

        data_item = {
            'id': idx,
            'inputImage1': os.path.relpath(img1_path, current_file_path),
            'inputImage2': os.path.relpath(img2_path, current_file_path),
            'reconstructionFile': os.path.relpath(model_paths[idx], current_file_path),
            'score': scores[idx]
        }
        data_list.append(data_item)

    data_json = json.dumps(data_list, indent=2)
    template_html = "template2.html"
    with open(template_html, 'r', encoding='utf-8') as f:
        html_template = f.read()
        html_output = html_template.replace(
            'const mockData = Array.from({ length: 1000 }, (_, i) => ({}));',
            f'const mockData = {data_json};'
        )
    
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
    output_html_path = os.path.join(output_html_dir, f'{timestamp_str}.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f'Generated {output_html_path} successfully.')


if __name__ == "__main__":

    """
    Run MASt3R on sample images.
    """
    # model = init_mast3r()
    # sparse_ga = run_reconstruction(model, ["/viscam/u/iamisaac/mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg", "/viscam/u/iamisaac/mast3r/assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg"], output_path="/viscam/u/iamisaac/mast3r/mast3r_outputs/")

    
    # mask1, mask2 = sparse_ga.masks
    # dinov2_similarity_score = get_dinov2_score("/viscam/u/iamisaac/mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg", "/viscam/u/iamisaac/mast3r/assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg", '/viscam/u/iamisaac/mast3r/mast3r_outputs/cache/corres_conf=desc_conf_subsample=8/7ee44180fd32b86548652184fe861e1c-b39877bbcbc90d4b6f93de98e8b85243.pth', mask1, mask2)
    
    # create_html([sparse_ga.img_paths[0]] *10, [sparse_ga.img_paths[1]] *10, [sparse_ga.recon_file_path] *10, [dinov2_similarity_score] *10, "/viscam/u/iamisaac/mast3r/htmls")

    """
    Performs 3D reconstruction using MASt3R on multiple query images.
    For each image, retrieves similar images from DataComp and runs reconstruction with varying thresholds.
    Logs results and 3D reconstructions to wandb for analysis and visualization.
    """
    model = init_mast3r()
    num_responses = 20
    threshold = 2.0
    json_file_path = 'query_img_paths.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    query_img_paths = [item['path'] for item in data['query_img_paths']]

    output_path = "mast3r_results"
    img1_paths = []
    img2_paths = []
    glb_model_paths = []
    dinov2_scores = []
    for query_img_path in query_img_paths:
        for i in range(0, num_responses):
            sparse_ga = run_mast3r_from_clip_retrieval(model, output_path=output_path, image_path=query_img_path,
                                        num_responses=num_responses, response_idxs=[i], indice_name="datacomp", display_images=True, min_conf_thr=threshold, include_query_image=True, image_cache_path=output_path)
            if not sparse_ga:
                continue
            img1_paths.append(sparse_ga.img_paths[0])
            img2_paths.append(sparse_ga.img_paths[1])
            glb_model_paths.append(sparse_ga.recon_file_path)
            dinov2_scores.append(sparse_ga.dinov2_score)
    create_html(img1_paths, img2_paths, glb_model_paths, dinov2_scores, "/viscam/u/iamisaac/mast3r/htmls")


    """
    Run MASt3R on CO3D dataset cars.
    """
    # car_dir = "/viscam/u/iamisaac/sfs/co3d_data/car/"
    # for d in os.listdir(car_dir):
    #     if os.path.isdir(os.path.join(car_dir, d)) and d[:2].isnumeric():
    #         dir_path = os.path.join(car_dir, d, "images")
    #         for file in os.listdir(dir_path)[0::50]:
    #             full_path = os.path.join(dir_path, file)
    #             run_mast3r_from_clip_retrieval(model, "/viscam/projects/sfs/mast3r_outputs/test1", image_path=full_path,
    #                                            num_responses=110, response_stride=100)
    #             break
