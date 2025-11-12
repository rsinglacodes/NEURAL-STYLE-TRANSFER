import os
import time
import uuid
import traceback
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ImageNet normalization for VGG (CPU tensors used for normalization step)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_image(path, max_size=256, shape=None):
    img = Image.open(path).convert('RGB')
    # Resize keeping aspect ratio
    max_dim = max(img.size)
    if max_dim > max_size and shape is None:
        scale = max_size / max_dim
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    if shape is not None:
        # shape provided as (H, W)
        img = img.resize((shape[1], shape[0]), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # 1,C,H,W
    # normalize (IMAGENET tensors are on CPU; we'll move to device later)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.matmul(features, features.transpose(1, 2))
    return gram / (c * h * w)

def denormalize(tensor):
    t = tensor.clone().detach()
    t = t * IMAGENET_STD + IMAGENET_MEAN
    t = t.clamp(0, 1)
    return t

def save_tensor_as_image(tensor, path):
    # tensor expected with shape 1,C,H,W on CPU
    t = denormalize(tensor.cpu().squeeze(0))
    img = transforms.ToPILImage()(t)
    img.save(path)

def get_vgg_features(x, model):
    # returns dict of named features
    layers = {
        0: 'conv1_1', 2: 'conv1_2',
        5: 'conv2_1', 7: 'conv2_2',
        10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3', 16: 'conv3_4',
        19: 'conv4_1', 21: 'conv4_2', 23: 'conv4_3', 25: 'conv4_4',
        28: 'conv5_1', 30: 'conv5_2', 32: 'conv5_3', 34: 'conv5_4'
    }
    features = {}
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx in layers:
            features[layers[idx]] = x
    return features

def neural_style_transfer(content_path, style_path, steps=100, content_weight=1, style_weight=1e5, max_size=256):
    print("NST: loading images...")
    content = load_image(content_path, max_size=max_size).to(device)
    style = load_image(style_path, shape=(content.size(2), content.size(3))).to(device)

    # initialize target as copy of content
    target = content.clone().requires_grad_(True).to(device)

    print("NST: loading model (may download weights on first run)...")
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    content_layer = 'conv4_2'
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # precompute features
    content_feats = get_vgg_features(content, vgg)
    style_feats = get_vgg_features(style, vgg)
    style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_layers}

    optimizer = optim.Adam([target], lr=0.02)

    print(f"NST: running optimization for {steps} steps...")
    start = time.time()
    for i in range(1, steps + 1):
        target_feats = get_vgg_features(target, vgg)

        c_loss = torch.mean((target_feats[content_layer] - content_feats[content_layer]) ** 2)

        s_loss = 0
        for layer in style_layers:
            t_grad = gram_matrix(target_feats[layer])
            s_grad = style_grams[layer]
            s_loss = s_loss + torch.mean((t_grad - s_grad) ** 2)

        total_loss = content_weight * c_loss + style_weight * s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % max(1, steps // 5) == 0 or i == 1:
            elapsed = time.time() - start
            print(f"Step {i}/{steps}  total={total_loss.item():.4f}  content={c_loss.item():.6f}  style={s_loss.item():.6f}  ({elapsed:.1f}s)")

    print("NST: done.")
    return target.detach()  # return tensor on device

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        content_file = request.files.get('content_image')
        style_file = request.files.get('style_image')
        if not content_file or not style_file:
            return render_template('index.html', error="Please provide both content and style images.")

        # sanitize original filenames
        orig_content_name = secure_filename(content_file.filename) if content_file.filename else None
        orig_style_name = secure_filename(style_file.filename) if style_file.filename else None

        # ensure we have fallback names
        orig_content_name = orig_content_name or "content.png"
        orig_style_name = orig_style_name or "style.png"

        # create unique filenames to avoid collisions / caching
        content_name = f"{uuid.uuid4().hex}_{orig_content_name}"
        style_name = f"{uuid.uuid4().hex}_{orig_style_name}"
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_name)
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_name)

        # save files
        content_file.save(content_path)
        style_file.save(style_path)

        # Quick check if user accidentally uploaded same file twice (compare file bytes length)
        try:
            if os.path.getsize(content_path) == os.path.getsize(style_path):
                warning = "Content and style file sizes are equal — make sure you didn't upload the same image twice."
            else:
                warning = None
        except Exception:
            warning = None

        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        # Run NST (blocking)
        try:
            # Changed weights and steps for better content preservation and convergence.
            # Increased content_weight from 1 to 5.
            # Decreased style_weight from 1e5 to 1e4 (or 1e3 for even stronger content).
            # Increased steps for better quality.
            
            # 💡 Suggested Settings (Try this first):
            # Settings to reproduce the strong style effect (like the Google image):
            result_tensor = neural_style_transfer(
                content_path, 
                style_path, 
                steps=200, 
                content_weight=0.1,     # Very low content weight
                style_weight=1e5,     # Very high style weight
                max_size=256
            )
            
            # Alternatively, for *very* strong content preservation (try if 5/1e4 isn't enough):
            # result_tensor = neural_style_transfer(content_path, style_path, steps=300, content_weight=10, style_weight=1e3, max_size=256)
            
            save_tensor_as_image(result_tensor.cpu(), result_path)
        except Exception as e:
        # ... (rest of the code)
            traceback.print_exc()
            return render_template('index.html', error=f"Generation failed: {str(e)}")

        content_url = url_for('static', filename=f"uploads/{content_name}") + f"?t={int(time.time())}"
        style_url = url_for('static', filename=f"uploads/{style_name}") + f"?t={int(time.time())}"
        result_url = url_for('static', filename=f"uploads/{result_filename}") + f"?t={int(time.time())}"

        return render_template('index.html',
                               content_image=content_url,
                               style_image=style_url,
                               result_image=result_url,
                               warning=warning)
    return render_template('index.html')

if __name__ == "__main__":
    # run with `python app.py`
    # Note: on CPU this can be slow — reduce steps or max_size if needed.
    app.run(debug=True)
