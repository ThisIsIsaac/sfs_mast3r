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

import mast3r.demo as demo
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import json
from pygltflib import GLTF2
import sfs_util
from torchmetrics.multimodal.clip_score import CLIPScore
import clip
import numpy as np
from datetime import datetime
import random

class Mast3r():
    def __init__(self, 
                 output_path="/viscam/projects/sfs/mast3r/mast3r_outputs", 
                 img_dir="/viscam/projects/sfs/mast3r/mast3r_outputs/imgs",
                 clip_keywords_file="/viscam/projects/sfs/mast3r/clip_keywords.json",
                 num_responses_per_query=5,
                 min_conf_threshold = 1.5,
                 matching_conf_threshold = 2.0,
                 device="cuda"
                 ):
        self.device=device
        self.img_dir = img_dir
        self.output_path = output_path
        self.existing_imgs = sfs_util.get_existing_images(img_dir);
        self.model = init_mast3r()
        self.min_conf_threshold = min_conf_threshold
        self.num_responses_per_query = num_responses_per_query
        self.html_dir = "/viscam/projects/sfs/mast3r/htmls"
        self.indice_name = "datacomp"
        self.matching_conf_threshold = matching_conf_threshold
        self.shared_intrinsics=False
        self.include_query_image=True
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=device)
        self.clip_keywords_file = clip_keywords_file

    def run_with_urls(self, urls, num_urls_to_use):
        img1_paths = []
        img2_paths = []
        glb_model_paths = []
        scores = []
        img_paths = []
        all_cache_paths = []
        for url in urls:
            try:
                img_path, _ = sfs_util.check_and_download_image(url, self.img_dir, self.existing_imgs)
                img_paths.append(img_path)
            except Exception as e:
                continue
            if len(img_paths) == num_urls_to_use: break
        if len(img_paths) < num_urls_to_use:
            raise ValueError(f"Not enough valid urls. Only {len(img_paths)} are valid and {num_urls_to_use} are requested.")

        for img_path in img_paths:
            results, cache_paths = self.run_mast3r_from_clip_retrieval(query_img_path=img_path, img_pair_mode="img1_pair")
            for sparse_ga, cache_path in zip(results, cache_paths):
                if not sparse_ga:
                    continue
                img1_paths.append(sparse_ga.img_paths[0])
                img2_paths.append(sparse_ga.img_paths[1])
                glb_model_paths.append(sparse_ga.recon_file_path)
                scores.append(sparse_ga.scores)
                all_cache_paths.extend(cache_path)
                
        if scores:
            sfs_util.create_html(img1_paths, img2_paths, glb_model_paths, scores, self.html_dir)
        self.delete_caches(all_cache_paths)
        
    
    def run_with_query_texts_file(self, query_text_json_file_path):
        if query_text_json_file_path:
            query_texts = []
            with open(query_text_json_file_path, 'r') as file:
                data = json.load(file)
                for class_name in data:
                    query_texts.extend(data[class_name])
                    
        self.run_with_query_texts(query_texts)
        
    def run_with_query_texts(self, query_texts):
        img1_paths = []
        img2_paths = []
        glb_model_paths = []
        scores = []
        all_cache_paths = []
        for query_text in query_texts:
            results, cache_paths = self.run_mast3r_from_clip_retrieval(query_text=query_text)
            for sparse_ga, cache_path in zip(results, cache_paths):
                if not sparse_ga:
                    continue
                img1_paths.append(sparse_ga.img_paths[0])
                img2_paths.append(sparse_ga.img_paths[1])
                glb_model_paths.append(sparse_ga.recon_file_path)
                scores.append(sparse_ga.scores)
                query_texts.append(query_text)
                all_cache_paths.append(cache_path)
        if scores:
            sfs_util.create_html(img1_paths, img2_paths, glb_model_paths, scores, self.html_dir, query_texts=query_texts)
        self.delete_caches(all_cache_paths)
        
    def delete_caches(self, cache_paths):
        for file in cache_paths:
            if not file: continue
            os.remove(file)
        
    def run_random_paris(self, num_pairs):
        img1_responses = sfs_util.send_query(
            existing_imgs=self.existing_imgs,
            image_cache_path=self.img_dir,
            query_text="high quality high resolution photo of an everyday object", 
            num_responses=num_pairs,
            indice_name=self.indice_name,
        )

        img1_paths = []
        img2_paths = []
        glb_model_paths = []
        scores = []
        query_texts = []
        all_cache_paths = []
        for img1_response in img1_responses:
            img1_path = img1_response["image_path"] 
            results, cache_paths = self.run_mast3r_from_clip_retrieval(query_img_path=img1_path, img_pair_mode="img1_pair")
            for sparse_ga, cache_path in zip(results, cache_paths):
                if not sparse_ga:
                    continue
                img1_paths.append(sparse_ga.img_paths[0])
                img2_paths.append(sparse_ga.img_paths[1])
                glb_model_paths.append(sparse_ga.recon_file_path)
                scores.append(sparse_ga.scores)
                query_texts.append("")
                all_cache_paths.extend(cache_path)
                
        if scores:
            sfs_util.create_html(img1_paths, img2_paths, glb_model_paths, scores, self.html_dir, query_texts=query_texts)
        self.delete_caches(all_cache_paths)

    def run_mast3r_from_clip_retrieval(self, query_text=None, query_img_path=None, response_idxs=None, img_pair_mode="all"):
        responses = sfs_util.send_query(
            existing_imgs=self.existing_imgs,
            image_cache_path=self.img_dir,
            image_path=query_img_path,
            query_text=query_text, 
            num_responses=self.num_responses_per_query,
            response_idxs=response_idxs,
            indice_name=self.indice_name,
            include_query_image=self.include_query_image
            
        )

        if img_pair_mode == "all":
            img_path_pairs = sfs_util.generate_all_unique_pairs(responses)
        elif img_pair_mode == "img1_pair":
            img_path_pairs = sfs_util.generate_paris(responses)
        else:
            raise ValueError("img_pair_mode should be either 'all' or 'img1_pair'")
        results = [] # A list of SparseGAs
        cache_paths = [] # a list of cached features to use be deleted after calculating scores
        if len(img_path_pairs) >= 1:
            for img_paths in img_path_pairs:
                sparse_ga = run_reconstruction(self.model, img_paths, min_conf_thr=self.min_conf_threshold, matching_conf_thr=self.matching_conf_threshold, shared_intrinsics=self.shared_intrinsics, output_path=self.output_path)
                if sparse_ga: print("Done!")
                if not sparse_ga.recon_file_path:
                    print(f"Matching failed for '{query_text}'!")
                    continue
                    # if image_path:
                    #     print(f"Matching failed for {image_path}!")
                    # continue

                mask1, mask2 = sparse_ga.masks
                scores = self.get_quality_scores(img_paths[0], img_paths[1], sparse_ga.recon_file_path, sparse_ga.cache_path, mask1, mask2)
                if not scores: continue # When there is zero confident pixel, retruns None
                sparse_ga.scores = scores
                if query_text:
                    sparse_ga.query_text = query_text
                results.append(sparse_ga)
                cache_paths.append(sparse_ga.cache_path)
            print(f"{len(results)} / {len(img_path_pairs)} reconstructions successfully performed.")
        else:
            print(f"Not enough images to run reconstruction!")
        return results, cache_paths

    def get_clip_scores(self, img_paths, texts, k):
        """Returns a list of dictionaries."""
        # Save CLIP scores to JSON
        clip_results = []
        bottom_k_scores = []

        for img_path in img_paths:
            scores = {}
            # img = torch.from_numpy(np.array(img)).to(self.device)
            raw_img = Image.open(img_path)
            img = self.clip_preprocess(raw_img).unsqueeze(0)
            img_features = self.clip_model.encode_image(img.to(self.device))
            for raw_text in texts:
                text = clip.tokenize(raw_text)

                text_features = self.clip_model.encode_text(text.to(self.device))

                # score_acc = 0.
                # sample_num = 0.
                logit_scale = self.clip_model.logit_scale.exp()
                    
                # normalize features
                real_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)
                fake_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)

                # calculate scores
                score = logit_scale * (fake_features * real_features).sum().round(decimals=4).detach().cpu().item()
                scores[raw_text] = float(score)
                
            result = {
                "image_path": img_path,
                "clip_scores": scores,
                "height": raw_img.height,
                "width": raw_img.width
            }
            clip_results.append(result)
            sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            lowest_k_dict = dict(sorted_pairs)
            bottom_k_scores.append(lowest_k_dict)

        # Save to JSON file with proper formatting with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"clip_scores/clip_scores_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(clip_results, f, indent=4)
        
        return bottom_k_scores

    def get_quality_scores(self, img1_path, img2_path, glb_file_path, cache_path, mask1, mask2, device='cuda'):
        """
        Produces:
        1. DinoV2 point-wise similarity score of mutual NNs
        2. Global conf score produced by MASt3R (sparse_ga.py)
            '(C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()'
        3. Point-wise conf score produced by MASt3R (sparse_ga.py)
            'torch.qrt([qonf11, qonf12] * [qonf21, qonf22])' of mutual NNs
        4. point-wise Mast3r feature distance of mutual NNs
        """
        scores = {}
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
        
        # Compute confidence mask
        """
        [
            [conf_score, sum of point-wise conf, number of mutual nn pairs],
            [xy1, xy2, point-wise conf]
        ]
        """
        cached_results = torch.load(cache_path)
        xy1, xy2, _ = cached_results[1]

        # Create boolean masks for valid points
        valid_mask1 = mask1[xy1[:, 1], xy1[:, 0]]
        valid_mask2 = mask2[xy2[:, 1], xy2[:, 0]]

        # Combine masks
        valid_mask = valid_mask1 & valid_mask2
        if valid_mask.sum() == 0:
            print(f"Zero confident pixels for images: ({img1_path}) and ({img2_path}) !!")
            return None

        # Apply the mask to get the filtered coordinates
        masked_xy1 = xy1[valid_mask]
        masked_xy2 = xy2[valid_mask]
            
        # Extract features for xy1 from reshaped_output1
        features1 = reshaped_output1[0, :, masked_xy1[:, 1], masked_xy1[:, 0]]   

        # Extract features for xy2 from reshaped_output2
        features2 = reshaped_output2[0, :, masked_xy2[:, 1], masked_xy2[:, 0]]

        similarity = F.cosine_similarity(features1, features2, dim=0)
        similarity_score = similarity.mean()
        
        
        # # Calculate clip scores aganst bad keywords
        # if self.clip_keywords_file:
        #     with open(self.clip_keywords_file, 'r') as file:
        #         clip_keywords = json.load(file)
        #     bad_clip_keywords = clip_keywords["bad"]
        #     assert(len(clip_keywords > 0))
        #     clip_scores = self.get_clip_scores([img1, img2], bad_clip_keywords, k=1)
            
        #     img1_clip_text, img1_clip_score = clip_scores[0].items()[0]
        #     img2_clip_text, img2_clip_score = clip_scores[1].items()[0]
        #     scores["img1_clip_text"] = img1_clip_text
        #     scores["img1_clip_score"] = img1_clip_score
        #     scores["img2_clip_text"] = img2_clip_text
        #     scores["img2_clip_score"] = img2_clip_score
            
        # Compute mast3r point-wise feature distance 
        feat11, feat21, feat22, feat12 = cached_results[2]
        assert feat11.shape == feat21.shape == feat22.shape == feat12.shape
        masked_feat11 = feat11[masked_xy1[:, 1], masked_xy1[:, 0], :]
        masked_feat21 = feat21[masked_xy2[:, 1], masked_xy2[:, 0], :]
        masked_feat12 = feat12[masked_xy1[:, 1], masked_xy1[:, 0], :]
        masked_feat22 = feat22[masked_xy2[:, 1], masked_xy2[:, 0], :]

        feature_distance = torch.norm(masked_feat11 - masked_feat21, dim=1) + torch.norm(masked_feat12 - masked_feat22, dim=1)
        feature_distance = feature_distance.mean()
        # Compute point-wise confidence score
        # qonf11, qonf21, qonf22, qonf12 = cached_results[3]
        # point_wise_conf_score = torch.sqrt(torch.sqrt(qonf11[valid_mask].mean() * qonf21[valid_mask].mean() * qonf22[valid_mask].mean() * qonf12[valid_mask].mean()))

        # scores["camera_angle"] = sfs_util.get_rotation_from_glb(glb_file_path)    
        scores["dinov2_score"] = similarity_score.item()
        scores["global_matching_score"] = cached_results[0][0]
        scores["point_wise_conf_score"] = cached_results[0][1]
        scores["feature_distance"] = feature_distance.item()
        scores["num_matching_pixels"] = cached_results[0][2]
        scores["num_confident_pixels"] = torch.sum(valid_mask).item()

        return scores


def init_mast3r(device='cuda'):
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    return model

def get_errors(pred_Rs, pred_Ts, gt_Rs, gt_Ts):
    R_err = []
    T_err = []
    for i in range(len(pred_Rs)):
        R_err.append(sfs_util.rotation_error(pred_Rs[i], gt_Rs[i]))
        T_err.append(sfs_util.translation_error(pred_Ts[i], gt_Ts[i]))
    pred_transform_R, pred_transform_T = sfs_util.compute_camera_transformation(pred_Rs[0], pred_Ts[0], pred_Rs[1], pred_Ts[1])
    gt_transform_R, gt_transform_T = sfs_util.compute_camera_transformation(gt_Rs[0], gt_Ts[0], gt_Rs[1], gt_Ts[1])
    R_transform_err = sfs_util.rotation_error(pred_transform_R, gt_transform_R)
    T_transform_err = sfs_util.translation_error(pred_transform_T, gt_transform_T)
    return R_transform_err, T_transform_err, R_err, T_err


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


if __name__ == "__main__":
    # with open("random_urls.json", "r") as file:
    #     urls = json.load(file)
    # mast3r = Mast3r(
    #     num_responses_per_query=15,
    #     min_conf_threshold = 1.5,
    #     matching_conf_threshold = 2.0,
    # )
    
    # with open("/viscam/projects/sfs/mast3r/bad_clip_keywords.json", 'r') as file:
    #     clip_keywords = json.load(file)
    # img1 = Image.open("/viscam/projects/sfs/mast3r/mast3r_outputs/imgs/ffe43230ae908c5a39d003b733661f4c.jpg")
    # img2 = Image.open("/viscam/projects/sfs/mast3r/mast3r_outputs/imgs/ff5b37b92672f48a11f98c94d15189d6.jpg")
    # mast3r.get_clip_scores([img1, img2], clip_keywords, 10)
    # mast3r.run_with_urls(urls, 200)
    
    # mast3r = Mast3r(
    #     min_conf_threshold = 1.0,
    #     matching_conf_threshold = 2.0
    # )
    # mast3r.run_with_query_texts_file("clip_query_texts.json")
    # mast3r.matching_conf_threshold = 1.0
    # mast3r.run_with_query_texts_file("clip_query_texts.json")
    mast3r = Mast3r()
    with open("/viscam/projects/sfs/mast3r/clip_scores/clip_keywords.json", 'r') as file:
        json_file = json.load(file)
        keywords = json_file["keywords"]
    img_paths = random.sample([entry.path for entry in os.scandir("/viscam/projects/sfs/mast3r/mast3r_outputs/random_imgs") if entry.is_file() and entry.name.endswith('.jpg')], k=500)
    
    mast3r.get_clip_scores(img_paths, keywords, 1)