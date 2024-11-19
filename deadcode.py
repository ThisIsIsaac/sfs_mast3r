        # if indice_name == "co3d":
        #     gt_R = [response["R"] for response in responses]
        #     gt_T = [response["T"] for response in responses]
        #     pred_R = [cam2w[:3, :3].cpu() for cam2w in sparse_ga.cam2w]
        #     pred_T = [cam2w[3:, 3:].cpu() for cam2w in sparse_ga.cam2w]
            # R_transform_err, T_transform_err, R_err, T_err = get_errors(pred_R, pred_T, gt_R, gt_T)



    """
    Run MASt3R on sample images.
    """
    # model = init_mast3r()
    # sparse_ga = run_reconstruction(model, ["/viscam/u/iamisaac/mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg", "/viscam/u/iamisaac/mast3r/assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg"], output_path="/viscam/u/iamisaac/mast3r/mast3r_outputs/")

    
    # mask1, mask2 = sparse_ga.masks
    # dinov2_similarity_score = get_quality_scores("/viscam/u/iamisaac/mast3r/assets/NLE_tower/1AD85EF5-B651-4291-A5C0-7BDB7D966384-83120-000041DADF639E09.jpg", "/viscam/u/iamisaac/mast3r/assets/NLE_tower/01D90321-69C8-439F-B0B0-E87E7634741C-83120-000041DAE419D7AE.jpg", '/viscam/u/iamisaac/mast3r/mast3r_outputs/cache/corres_conf=desc_conf_subsample=8/7ee44180fd32b86548652184fe861e1c-b39877bbcbc90d4b6f93de98e8b85243.pth', mask1, mask2)
    
    # create_html([sparse_ga.img_paths[0]] *10, [sparse_ga.img_paths[1]] *10, [sparse_ga.recon_file_path] *10, [dinov2_similarity_score] *10, "/viscam/u/iamisaac/mast3r/htmls")

    """
    Performs 3D reconstruction using MASt3R on multiple query images.
    For each image, retrieves similar images from DataComp and runs reconstruction with varying thresholds.
    Logs results and 3D reconstructions to wandb for analysis and visualization.
    """
    # model = init_mast3r()
    # num_responses = 20
    # threshold = 2.0
    # json_file_path = 'query_img_paths.json'
    # with open(json_file_path, 'r') as file:
    #     data = json.load(file)
    # query_img_paths = [item['path'] for item in data['query_img_paths']]

    # output_path = "mast3r_results"
    # img1_paths = []
    # img2_paths = []
    # glb_model_paths = []
    # scores = []

    # for query_img_path in query_img_paths:
    #     for i in range(0, num_responses):
    #         results = run_mast3r_from_clip_retrieval(model, output_path=output_path, image_path=query_img_path,
    #                                     num_responses=num_responses, response_idxs=[i], indice_name="datacomp", min_conf_thr=threshold, include_query_image=True, image_cache_path=output_path)
    #         sparse_ga = results[0]
    #         if not sparse_ga:
    #             continue
    #         img1_paths.append(sparse_ga.img_paths[0])
    #         img2_paths.append(sparse_ga.img_paths[1])
    #         glb_model_paths.append(sparse_ga.recon_file_path)
    #         scores.append(sparse_ga.scores)

    # create_html(img1_paths, img2_paths, glb_model_paths, scores, "/viscam/u/iamisaac/mast3r/htmls")


    """
    Run MASt3R on CO3D dataset cars.
    """
    # model = init_mast3r()
    # img1_paths = []
    # img2_paths = []
    # glb_model_paths = []
    # scores = []
    # car_dir = "/viscam/u/iamisaac/sfs/co3d_data/car/"
    # for d in os.listdir(car_dir):
    #     if os.path.isdir(os.path.join(car_dir, d)) and d[:2].isnumeric():
    #         dir_path = os.path.join(car_dir, d, "images")
    #         for file in os.listdir(dir_path)[0::50]:
    #             full_path = os.path.join(dir_path, file)
    #             results = run_mast3r_from_clip_retrieval(model, "/viscam/projects/sfs/mast3r_outputs/test1", image_path=full_path,
    #                                            num_responses=110,)
    #             sparse_ga = results[0]
    #             img1_paths.append(sparse_ga.img_paths[0])
    #             img2_paths.append(sparse_ga.img_paths[1])
    #             glb_model_paths.append(sparse_ga.recon_file_path)
    #             scores.append(sparse_ga.scores)
    # create_html(img1_paths, img2_paths, glb_model_paths, scores, "/viscam/u/iamisaac/mast3r/htmls")
    