from mmcv.image import imread
from argparse import ArgumentParser
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

model_cfg = 'configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py'
# 'configs/body_2d_keypoint/topdown_heatmap/jhmdb/td-hm_res50-2deconv_8xb64-40e_jhmdb-sub2-256x256.py'
# 'configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py'
#configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py
#configs/body_2d_keypoint/rtmpose/body8/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py
ckpt = 'cpt/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth'
# 'cpt/res50_2deconv_jhmdb_sub2_256x256-f63af0ff_20201122.pth'
# 'cpt/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth'
# 'cpt/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
# cpt/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth
device = 'cuda'

# init model
model = init_model(model_cfg, ckpt, device=device)

img_in_dir = 'demo/mybody/'
img_name = '正面.jpg'
img_out_dir = 'demo/mybody/res/'
img_path = img_in_dir + img_name
out_img_path = img_out_dir + img_path
# inference on a single image
batch_results = inference_topdown(model, img_path)

parser = ArgumentParser()

args = parser.parse_args()
args.side = 1 #1正面 2 左侧面 3 右侧面 4背面
args.show_lines = True
args.radius = 7
args.alpha = 1
args.thickness = 5
args.skeleton_style = 'mmpose'
args.img = img_path
args.kpt_thr = 0.3
args.draw_heatmap = False
args.show_kpt_idx = True
args.show = False
args.out_file = out_img_path
# init visualizer
model.cfg.visualizer.radius = args.radius
model.cfg.visualizer.alpha = args.alpha
model.cfg.visualizer.line_width = args.thickness

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(
    model.dataset_meta, skeleton_style=args.skeleton_style, side = args.side, show_lines = args.show_lines)

# inference a single image
batch_results = inference_topdown(model, args.img)

pred_instances = batch_results[0].pred_instances


results = merge_data_samples(batch_results, args.side)

# show the results
img = imread(args.img, channel_order='rgb')
visualizer.add_datasample(
    'result',
    img,
    data_sample=results,
    draw_gt=False,
    draw_bbox=False,
    kpt_thr=args.kpt_thr,
    draw_heatmap=args.draw_heatmap,
    show_kpt_idx=args.show_kpt_idx,
    skeleton_style=args.skeleton_style,
    show=args.show,
    out_file=args.out_file,
    side=args.side,)