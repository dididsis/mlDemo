import torch
import numpy as np
from PIL import Image
import cv2

from torchvision import transforms
from skimage.morphology import remove_small_holes
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize
from networks.vision_transformer import SwinUnet
from config import get_config

class Args:
    pass

args = Args()
# 必要な引数をここでハードコード
args.root_path = "."
args.dataset = "datasets"
args.list_dir = f'.'
args.num_classes = 2  # 必要に応じて変更
args.output_dir = "."
args.max_iterations = 15000
args.max_epochs = 50
args.batch_size = 8
args.n_gpu = 1
args.deterministic = 1
args.base_lr = 5e-5
args.img_size = 224
args.seed = 123
args.cfg = "swin_tiny_patch4_window7_224_lite.yaml"  # cfgファイルパスを指定

# 追加オプションは必要に応じてハードコード
args.opts = None
args.zip = False
args.cache_mode = 'part'
args.resume = None
args.accumulation_steps = None
args.use_checkpoint = False
args.amp_opt_level = 'O1'
args.tag = None
args.eval = False
args.throughput = False
args.n_class = 2
args.num_workers = 2
args.eval_interval = 1

config = get_config(args)

num_classes = 2
model = SwinUnet(config, img_size=224, num_classes=num_classes).cuda()
model_weights_path = "best_model.pth"

state_dict = torch.load(model_weights_path)
model.load_state_dict(state_dict=state_dict)
model.eval()

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def pred(image):
    input_tensor = transform(image).unsqueeze(0)

    if device.type  == 'cuda': 
        input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs= model(input_tensor)

    pred=torch.softmax(outputs, dim=1)
    pred_mask=torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    #print(np.unique(pred_mask))
    pred_mask=(pred_mask>0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)

    min_area = 50  # 小さなノイズとみなす面積の閾値

    # stats[i, cv2.CC_STAT_AREA] が連結成分 i の画素数
    for i in range(1, num_labels):  # 0番目は背景ラベル
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            # 面積が閾値未満なら除去（0にする）
            pred_mask[labels == i] = 0
    kernel = np.ones((3,3), np.uint8)

    opened = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 好みに応じてiteration回数やカーネルサイズを変える
    pred_mask = closed  # 最終的なマスク
    pred_mask = remove_small_holes(pred_mask.astype(bool), area_threshold=100).astype(np.uint8)
    pred_mask = pred_mask * 255

    return pred_mask

def pred_large_image(original_image, patch_size=224):
    w, h = original_image.size

    nx = (w + patch_size - 1) // patch_size
    ny = (h + patch_size - 1) // patch_size

    full_mask = np.zeros((h, w), dtype = np.uint8)

    for y_idx in range(ny):
        for x_idx in range(nx):
            x0 = x_idx * patch_size
            y0 = y_idx * patch_size
            x1 = min(x0 + patch_size, w)
            y1 = min(y0 + patch_size, h)

            tile_w = x1 - x0
            tile_h = y1 - y0

            tile = original_image.crop((x0,y0,x1,y1))

            if tile_w < patch_size or tile_h < patch_size:
                padded_img = Image.new("RGB", (patch_size, patch_size), color=(255, 255, 255))
                padded_img.paste(tile, (0, 0))
                tile = padded_img

            tile_mask_224 = pred(tile)
            tile_mask_cropped = tile_mask_224[0:tile_h, 0:tile_w]

            full_mask[y0:y1, x0:x1] = tile_mask_cropped
        
    return full_mask

def eval(pred):
    skeleton = skeletonize(pred > 0)
    skeleton_pixels = np.argwhere(skeleton)

    # skeleton は True/False のbool画像なので、0/1にしたければ
    skeleton_mask = skeleton.astype(np.uint8)
    a, distance = medial_axis(pred, return_distance=True)
    width_values = 2.0 * distance[skeleton_pixels[:, 0], skeleton_pixels[:, 1]]

    mean_width = np.mean(width_values)
    std_width = np.std(width_values)

    print("平均値:", mean_width)
    print("標準偏差:", std_width)

    k = 3
    lower_bound = mean_width - k * std_width
    upper_bound = mean_width + k * std_width

    # フィルタリング
    filtered_values = width_values[
        (width_values >= lower_bound) & (width_values <= upper_bound)
    ]
    # 幅を入れるための画像を準備 (float型)
    width_map = np.zeros_like(pred, dtype=np.float32)

    # スケルトン画素だけ幅値を格納
    for (y, x), w in zip(skeleton_pixels, filtered_values):
        width_map[y, x] = w

    return [skeleton_mask * 255, filtered_values, width_map, mean_width, std_width]