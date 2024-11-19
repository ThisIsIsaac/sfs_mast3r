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

# visualize a few matches
import torch
from matplotlib import pyplot as plt


import numpy as np
import requests
from PIL import Image
import base64
import io
import ast
import copy
from clip_retrieval.clip_back import download_image 

from urllib.parse import urlparse

import json
from datetime import datetime
import shutil
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as Rscipy
import hashlib
from torchvision import transforms

MIN_IMG_SIZE = 64

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


def get_rotation_from_glb(glb_file_path):
    gltf = GLTF2().load(glb_file_path)

    # Extract camera nodes
    nodes = gltf.nodes
    camera_nodes = []
    for idx, node in enumerate(nodes):
        if node.camera is not None:
            camera_nodes.append((idx, node))

    if len(camera_nodes) < 2:
        raise ValueError("Less than two camera nodes found in the GLB file.")

    # Function to compute the transformation matrix of a node
    def get_node_matrix(node):
        matrix = np.eye(4)
        if node.translation is not None:
            T = np.eye(4)
            T[:3, 3] = node.translation
            matrix = matrix @ T
        if node.rotation is not None:
            R = Rscipy.from_quat([
                node.rotation[0],  # x
                node.rotation[1],  # y
                node.rotation[2],  # z
                node.rotation[3],  # w
            ]).as_matrix()
            R_matrix = np.eye(4)
            R_matrix[:3, :3] = R
            matrix = matrix @ R_matrix
        if node.scale is not None:
            S = np.diag([node.scale[0], node.scale[1], node.scale[2], 1])
            matrix = matrix @ S
        if node.matrix is not None and any(node.matrix):
            matrix = np.array(node.matrix).reshape(4, 4)
        return matrix

    # Get transformation matrices of the first two camera nodes
    idx1, node1 = camera_nodes[0]
    idx2, node2 = camera_nodes[1]
    matrix1 = get_node_matrix(node1)
    matrix2 = get_node_matrix(node2)

    # Compute rotation between the two cameras
    R1 = matrix1[:3, :3]
    R2 = matrix2[:3, :3]
    R_rel = R2 @ R1.T
    r_rel = Rscipy.from_matrix(R_rel)
    angle_axis = r_rel.as_rotvec()
    angle = np.linalg.norm(angle_axis)
    axis = angle_axis / angle if angle != 0 else angle_axis
    return angle

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

        # If content-type check passes, try to get the image content incrementally
        from PIL import ImageFile

        response = requests.get(url, stream=True, timeout=5)

        parser = ImageFile.Parser()

        # Read the data incrementally
        try:
            for chunk in response.iter_content(chunk_size=1024):
                parser.feed(chunk)
                if parser.image:
                    width, height = parser.image.size
                    # Check if image dimensions are large enough.
                    return width > MIN_IMG_SIZE and height > MIN_IMG_SIZE
            # If we reach here, the image size could not be determined
            return False
        except Exception:
            # If an error occurs (e.g., invalid image), return False
            return False


    except Exception as e:
        print(f"Error checking URL {url}: {str(e)}")
        return False

def send_query(existing_imgs, response_idxs=None, image_path=None, query_text=None, num_responses = 5, indice_name="co3d", image_cache_path=None, include_query_image=False):
    data = {
        "num_images": num_responses*10,
        "num_result_ids": num_responses*10,
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
    results = parse_response(response, response_idxs=response_idxs, indice_name=indice_name, num_responses=num_responses, img_dir=image_cache_path, include_query_image=include_query_image, query_image_path=image_path, existing_imgs=existing_imgs)
    return results

def url_to_filename(url):
    """Hash the URL to create a unique filename."""
    hash_object = hashlib.md5(url.encode('utf-8'))
    filename_hash = hash_object.hexdigest()
    return filename_hash

def get_existing_images(directory):
    """Create a set of existing image filenames in the directory."""
    # List of common image file extensions
    extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    existing_files = set()
    for filename in os.listdir(directory):
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            if ext.lower() in extensions:
                existing_files.add(filename)
    return existing_files

def image_exists(filename_hash, existing_files):
    """Check if an image with the given hash already exists in the directory."""

    # Check if any file with the filename_hash and any of the extensions exists
    filename = f"{filename_hash}.jpg"
    if filename in existing_files:
        return True
    return False

def check_and_download_image(url, directory, existing_files):
    """Download the image from the URL if it doesn't already exist."""
    filename_hash = url_to_filename(url)
    filename = f"{filename_hash}.jpg"
    filepath = os.path.join(directory, filename)
    if image_exists(filename_hash, existing_files):
        image = Image.open(filepath)
    else:
        image_data = download_image(url)
        if isinstance(image_data, io.BytesIO):
            image = Image.open(image_data).resize((256, 256))
        else:
            image = Image.open(io.BytesIO(image_data)).resize((256, 256))
        image.save(filepath)
        existing_files.add(filename)
    return filepath, np.array(image)

def parse_response(response, indice_name, num_responses, existing_imgs, response_idxs=None, img_dir=None, include_query_image=False, query_image_path=None):
    seen_urls = set()
    results = response.json()
    if response_idxs: results = [results[idx] for idx in response_idxs]
    parsed_results = []
    if query_image_path and include_query_image:
        parsed_results.append({
            "image_path": query_image_path,
            "image": np.array(Image.open(query_image_path))
        })
    for result in results:
        if result["url"] in seen_urls:
            continue
        seen_urls.add(result["url"])
        parsed_result = copy.deepcopy(result)
        if "image" in result:
            image = Image.open(io.BytesIO(base64.b64decode(result["image"])))
            parsed_result["image"] = np.array(image)
        elif "url" in result:
            if is_valid_image_url(result["url"]):
                try:
                    parsed_result["image_path"], parsed_result["image"] = check_and_download_image(result["url"], img_dir, existing_imgs)
                except Exception as e:
                    print(f"Error downloading or saving image from url: {result['url']}")
                    print(f"Error details: {str(e)}")
                    continue
            else:
                print(f"Invalid or non-image URL: {result['url']}")
                continue

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
        if len(parsed_results) == num_responses:
            break
    return parsed_results

def generate_paris(responses):
    img_paths = [response["image_path"] for response in responses]
    imgs = [response["image"] for response in responses]
    if not img_paths or not imgs: return []
    unique_pairs = []
    img1_path = img_paths[0]
    img1 = imgs[0]
    n = len(img_paths)
    # Iterate over each path
    for i in range(1, len(img_paths)):
        if img_paths[i] == img1_path or np.array_equal(imgs[i], img1):
            continue
        unique_pairs.append([img1_path, img_paths[i]])
    return unique_pairs

def generate_all_unique_pairs(responses):
    img_paths = [response["image_path"] for response in responses]
    imgs = [response["image"] for response in responses]
    
    unique_pairs = []
    n = len(img_paths)
    # Iterate over each path
    for i in range(n):
        # Pair the current path with all subsequent paths to avoid duplicates
        for j in range(i + 1, n):
            # Check if the paths or images of the two pairs are identical; if so, skip this pair
            if img_paths[i] == img_paths[j] or np.array_equal(imgs[i], imgs[j]):
                continue
            unique_pairs.append([img_paths[i], img_paths[j]])
    return unique_pairs


def aggregate_scores(scores):
    # Extract all keys
    keys = list(scores[0].keys())

    # Collect values for each key across all dictionaries
    key_values = {key: [score[key] for score in scores] for key in keys}
    num_scores = len(scores)

    import numpy as np
    from scipy.stats import rankdata

    for key in keys:
        values = np.array(key_values[key])

        # Calculate percentiles
        ranks = rankdata(values, method='average')
        percentiles = (ranks - 1) / max(1, (num_scores - 1)) * 100  # Convert ranks to percentiles

        # Calculate ascending order indices
        asc_order = np.argsort(values)
        asc_ranks = np.empty_like(asc_order)
        asc_ranks[asc_order] = np.arange(num_scores)

        # Calculate descending order indices
        desc_order = np.argsort(-values)
        desc_ranks = np.empty_like(desc_order)
        desc_ranks[desc_order] = np.arange(num_scores)

        # Update each dictionary with the new keys
        for idx, score in enumerate(scores):
            score[f'{key}_percentile'] = percentiles[idx]
            score[f'{key}_asc_rank'] = int(asc_ranks[idx])
            score[f'{key}_desc_rank'] = int(desc_ranks[idx])

def create_html(img1_paths, img2_paths, model_paths, scores, output_html_dir, query_texts=[]):
    data_list = []
    score_names = list(scores[0].keys())
    aggregate_scores(scores) #in-place operation
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
            'scores': scores[idx],
            'score_names': score_names,
        }
        if query_texts and query_texts[idx]: data_item['query_text'] = query_texts[idx]
        data_list.append(data_item)

    # Save the data to a JSON file
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
    json_filename = f'data_{timestamp_str}.json'
    json_filepath = os.path.join(output_html_dir, json_filename)
    with open(json_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=2)

    
    template_html = os.path.join(output_html_dir,"template.html")
    with open(template_html, 'r', encoding='utf-8') as f:
        html_template = f.read()
        html_output = html_template.replace(
            "const dataFilePath = 'path.json';",
            f'const dataFilePath = "{json_filename}";'
        )

    output_html_path = os.path.join(output_html_dir, f'{timestamp_str}.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    print(f'Generated {output_html_path} successfully.')

def pil_to_tensor(image_path, normalize=True):
    # Open the image file
    img = Image.open(image_path).convert('RGB')

    
    # Define the transformation pipeline
    transform_list = [
        transforms.ToTensor()
    ]
    
    # Add normalization if requested
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                   std=[0.26862954, 0.26130258, 0.27577711]))
    
    transform = transforms.Compose(transform_list)
    
    # Apply the transformation
    tensor = transform(img)
    
    return tensor
