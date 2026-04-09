import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import auc
from torch.utils.data import Dataset
from torchvision import datasets

def ensure_prob_matrix(prob, class_order):
    prob = np.asarray(prob, dtype=float)
    class_order = np.asarray(class_order)

    if prob.ndim == 1:
        if len(class_order) != 2:
            raise ValueError("1D prob is only supported for binary (2 classes).")
        p_pos = prob
        return np.column_stack([1.0 - p_pos, p_pos])

    if prob.ndim == 2:
        if prob.shape[1] != len(class_order):
            raise ValueError(
                f"prob has {prob.shape[1]} columns but class_order has {len(class_order)}."
            )
        return prob

    raise ValueError("prob must be shape (n,) or (n,c).")

def align_proba_to_class_order(prob, model_class_order, target_class_order):
    """
    Align probability matrix columns to match a target class order.

    Parameters
    ----------
    prob : array-like, shape (n_samples, n_classes)
        Probability matrix with columns in model_class_order
    model_class_order : array-like
        Current order of classes (e.g., model.classes_ for sklearn)
    target_class_order : array-like
        Desired order of classes

    Returns
    -------
    np.ndarray
        Probability matrix with columns reordered to match target_class_order

    Examples
    --------
    prob_aligned = align_proba_to_class_order(model.predict_proba(x), model.classes_, [0, 1, 2])
    """
    prob = np.asarray(prob)
    model_class_order = list(model_class_order)
    target_class_order = list(target_class_order)

    # Find the index mapping
    idx = [model_class_order.index(c) for c in target_class_order]

    return prob[:, idx]


class ScaledLinearHead(nn.Module):
    """
    Linear head that optionally applies the same scaler as sklearn models.
    Scaling is in the forward pass so Grad-CAM path matches sklearn preprocessing.
    """

    def __init__(self, in_dim, n_classes, scaler=None, eps=1e-12):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
        self.has_scaler = scaler is not None

        if self.has_scaler:
            mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            # avoid division by zero
            scale = torch.clamp(scale, min=eps)
            self.register_buffer('mean', mean)
            self.register_buffer('scale', scale)

    def forward(self, feats):
        if self.has_scaler:
            feats = (feats - self.mean) / self.scale
        return self.linear(feats)


class CAMModel(nn.Module):
    """
    Simple wrapper
    """

    def __init__(self, feature_extractor, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.head(feats)


class GradCAM:
    """
    Grad-CAM for a CAMModel.

    By default, attempts to hook into `feature_extractor.layer4[-1].conv2` for ResNet18/34.
    Provide `target_layer` if different.
    """

    def __init__(self, cam_model, target_layer=None):
        self.model = cam_model

        if target_layer is None:
            fe = cam_model.feature_extractor
            if hasattr(fe, 'layer4'):
                target_layer = fe.layer4[-1].conv2
            else:
                raise ValueError('Cannot auto-detect target layer. Provide target_layer')

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self):
        if getattr(self, '_fwd_handle', None) is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None

        if getattr(self, '_bwd_handle', None) is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def _save_activation(self, _module, _inp, out):
        self.activations = out

    def _save_gradient(self, _module, _grad_inp, grad_out):
        self.gradients = grad_out[0]

    @torch.no_grad()
    def predict_classes(self, images, device, batch_size=64):
        """
        Predict argmax class for each image.

        Parameters
        ----------
        images :
            Input tensor (N, C, H, W)
        device :
            Torch device
        batch_size :
            Batch size for forward passes

        Returns
        -------
        np.ndarray
            Predicted class
        """
        self.model.eval()
        preds = []
        for i in range(0, len(images), batch_size):
            x = images[i: i + batch_size].to(device, non_blocking=True)
            logits = self.model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def cam_single(self, image, target_class=None, device=None):
        """
        Compute Grad-CAM for a single image.

        Parameters
        ----------
        image :
            Tensor of shape (C, H, W) or (1, C, H, W)
        target_class :
            Class to explain. If None, uses model argmax
        device :
            Device to run on. If None, inferred from model parameters

        Returns
        -------
        np.ndarray
            Normalized heatmap (H, W) in [0, 1]
        """
        if device is None:
            device = next(self.model.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(device, non_blocking=True)
        image.requires_grad_(True)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        logits = self.model(image)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('GradCAM hooks did not capture activations or gradients.')

        # Weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = nnf.relu(cam)

        cam = nnf.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        mn, mx = float(cam.min()), float(cam.max())
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)

        return cam


def train_cam_model(feature_extractor, images, labels, scaler=None,
                    n_classes=None, device=None,
                    epochs=15, lr=1e-3, batch_size=64, verbose=True):
    """
    Train a linear head (with optional scaler) on top of a frozen feature extractor.
    Uses true labels.

    Parameters
    ----------
    feature_extractor :
        Torch feature extractor (e.g., ResNet with removed classifier)
    images :
        Image tensor (N, C, H, W)
    labels :
        Class labels, length N
    scaler :
        sklearn-like StandardScaler to embed into the head forward pass
    n_classes :
        Number of classes. If None, inferred from unique labels
    device :
        Torch device. If None, inferred from feature_extractor parameters
    epochs, lr, batch_size :
        Training hyperparameters
    verbose :
        Print progress

    Returns
    -------
    CAMModel
        Frozen feature extractor and trained head
    """

    if device is None:
        device = next(feature_extractor.parameters()).device

    feature_extractor.eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = int(len(np.unique(labels)))

    # Extract features once
    if verbose:
        print('Extracting raw features for CAM training...')

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(x).cpu().numpy())
    feats = np.vstack(feats_list)

    in_dim = feats.shape[1]
    head = ScaledLinearHead(in_dim, n_classes, scaler=scaler).to(device)
    cam_model = CAMModel(feature_extractor, head).to(device)

    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size, shuffle=True
    )

    opt = torch.optim.Adam(cam_model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f'Training CAM head for {epochs} epochs...')

    for ep in range(epochs):
        cam_model.head.train()
        tot_loss, correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = cam_model.head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tot_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f'Epoch {ep + 1:02d}/{epochs}: loss={tot_loss / len(loader):.4f}, acc={100 * correct / total:.2f}%')

    cam_model.eval()
    return cam_model


def blur_images_gaussian(images, ksize=31, sigma=7.0):
    """
    Applies Gaussian blur to a batch of images (N, C, H, W) using separable conv.

    Parameters
    ----------
    images :
        Input images tensor (N, C, H, W)
    ksize :
        Kernel size. If even, it will be incremented by 1 to keep it odd
    sigma :
        Gaussian sigma

    Returns
    -------
    torch.Tensor
        Blurred images, same shape as input
    """
    if ksize % 2 == 0:
        ksize += 1

    device = images.device
    dtype = images.dtype

    # Gaussian kernel
    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # Separable kernels
    g_x = g.view(1, 1, 1, ksize).repeat(images.shape[1], 1, 1, 1)
    g_y = g.view(1, 1, ksize, 1).repeat(images.shape[1], 1, 1, 1)

    pad = ksize // 2
    out = nnf.conv2d(images, g_x, padding=(0, pad), groups=images.shape[1])
    out = nnf.conv2d(out, g_y, padding=(pad, 0), groups=images.shape[1])
    return out


def compute_gradcam_maps(images, cam_model, device=None, batch_pred=64, verbose=True):
    """
    Compute Grad-CAM importance maps for a batch of images.

    Parameters
    ----------
    images :
        Image tensor (N, C, H, W)
    cam_model :
        CAMModel used for Grad-CAM
    device :
        Torch device. If None, inferred from cam_model parameters
    batch_pred :
        Batch size used for predicting target classes
    verbose :
        Print progress

    Returns
    -------
    np.ndarray
        Importance maps, shape (N, H, W), dtype float32.
    """
    if device is None:
        device = next(cam_model.parameters()).device

    gradcam = GradCAM(cam_model)

    if verbose:
        print('Predicting target classes for Grad-CAM...')
    targets = gradcam.predict_classes(images, device=device, batch_size=batch_pred)

    if verbose:
        print('Computing Grad-CAM maps...')
    maps = []
    for i in range(len(images)):
        maps.append(gradcam.cam_single(images[i:i + 1], target_class=int(targets[i]), device=device))
        if verbose and (i + 1) % 100 == 0:
            print(f'{i + 1}/{len(images)} maps')

    gradcam.close()
    return np.asarray(maps, dtype=np.float32)


def precompute_patch_rankings(importance_maps, patch_size=32):
    """
    Convert per-pixel importance maps into per-image patch rankings.

    Parameters
    ----------
    importance_maps :
        Array of shape (N, H, W)
    patch_size:
        Size of square patches

    Returns
    -------
    rankings :
        List of length N with arrays of patch indices sorted by descending importance
    meta :
        PatchMeta describing the grid
    """
    n, h, w = importance_maps.shape
    n_ph = h // patch_size
    n_pw = w // patch_size

    patch_coords = []
    for ph in range(n_ph):
        for pw in range(n_pw):
            y0 = ph * patch_size
            x0 = pw * patch_size
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            patch_coords.append((y0, y1, x0, x1))

    rankings = []
    for i in range(n):
        imp = importance_maps[i]
        scores = np.array([imp[y0:y1, x0:x1].mean() for (y0, y1, x0, x1) in patch_coords], dtype=np.float32)
        rankings.append(np.argsort(scores)[::-1])

    meta = {
        'patch_size': patch_size,
        'patch_coords': patch_coords,
        'total_patches': len(patch_coords),
        'n_patches_h': n_ph,
        'n_patches_w': n_pw,
    }
    return rankings, meta


def apply_importance_masking(images, patch_rankings, patch_meta, fraction_to_mask,
                             mask_strategy='most_important',
                             mask_value=0.0,
                             baseline='constant',
                             blur_ksize=31, blur_sigma=7.0):
    """
    Mask a fraction of the image area using patch importance rankings.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    patch_rankings :
        Per-image arrays of patch indices sorted by descending patch importance
    patch_meta :
        Produced by `precompute_patch_rankings`
    fraction_to_mask :
        Fraction of total pixels to mask in [0, 1]
    mask_strategy :
        'most_important' masks top-ranked patches, supports adding new strategies later
    mask_value :
        Constant value used if baseline='constant'
    baseline :
        - 'constant': fill masked area with mask_value
        - 'blur': replace masked area with blurred content
    blur_ksize, blur_sigma :
        Parameters for Gaussian blur baseline

    Returns
    -------
    torch.Tensor
        Masked images, same shape as input
    """
    out = images.clone()
    n, c, h, w = out.shape

    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)
    else:
        blurred = None

    patch_size = patch_meta['patch_size']
    patch_pixels = patch_size * patch_size
    total_pixels = h * w

    pixels_to_mask = int(fraction_to_mask * total_pixels)
    k = pixels_to_mask // patch_pixels
    k = min(k, patch_meta['total_patches'])
    if k <= 0:
        return out

    coords = patch_meta['patch_coords']

    for i in range(n):
        order = patch_rankings[i]
        if mask_strategy == 'most_important':
            chosen = order[:k]
        else:
            raise ValueError(f'Unknown mask_strategy: {mask_strategy}')

        for idx in chosen:
            y0, y1, x0, x1 = coords[int(idx)]
            if baseline == 'blur':
                out[i, :, y0:y1, x0:x1] = blurred[i, :, y0:y1, x0:x1]
            else:
                out[i, :, y0:y1, x0:x1] = mask_value

    return out


def apply_patch_occlusion(images, num_patches, patch_size=32, random_seed=None,
                          mask_value=0.0,
                          baseline='constant',
                          blur_ksize=31, blur_sigma=7.0):
    """
    Random patch masking.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    num_patches :
        Number of random patches to mask per image
    patch_size :
        Square patch size
    random_seed :
        If provided, seeds torch and numpy for reproducibility
    mask_value :
        Constant fill value used when baseline='constant'
    baseline :
        - 'constant': fill masked area with mask_value
        - 'blur': replace masked area with blurred content
    blur_ksize, blur_sigma:
        Parameters for Gaussian blur baseline

    Returns
    -------
    torch.Tensor
        Masked images, same shape as input
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    out = images.clone()
    n, c, h, w = out.shape
    if num_patches <= 0:
        return out

    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)
    else:
        blurred = None

    for i in range(n):
        for _ in range(num_patches):
            y0 = random.randint(0, h - patch_size)
            x0 = random.randint(0, w - patch_size)
            if baseline == 'blur':
                out[i, :, y0:y0 + patch_size, x0:x0 + patch_size] = blurred[i, :, y0:y0 + patch_size,
                                                                    x0:x0 + patch_size]
            else:
                out[i, :, y0:y0 + patch_size, x0:x0 + patch_size] = mask_value
    return out


def extract_features_from_images(images, feature_extractor, pca=None, scaler=None,
                                 device=None, batch_size=64):
    """
    Extract features from images using a torch feature extractor, optionally apply PCA and scaling.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    feature_extractor :
        Torch module mapping images -> features (N, D)
    pca :
        sklearn-like object with `.transform(x)` or None
    scaler :
        sklearn-like object with `.transform(x)` or None
    device :
        Device for feature extraction. If None, inferred from feature_extractor
    batch_size :
        Batch size for extraction

    Returns
    -------
    np.ndarray
        Feature matrix after optional PCA and scaling
    """

    feature_extractor.eval()
    if device is None:
        device = next(feature_extractor.parameters()).device

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(batch).cpu().numpy())
    x = np.vstack(feats_list)

    if pca is not None:
        x = pca.transform(x)
    if scaler is not None:
        x = scaler.transform(x)

    return x


def get_predictions_from_features(features, model, model_class_order, class_order,
                                  model_type='sklearn', device=None, batch_size=64):
    """
    Get class probabilities from a model given feature vectors and align them to `class_order`.

    Parameters
    ----------
    features :
        Feature matrix (N, D)
    model :
        sklearn model with predict_proba or torch module producing logits
    model_class_order :
        Class labels order as produced by the model (e.g., sklearn `model.classes_`)
    class_order :
        Desired canonical class order
    model_type :
        'sklearn' or 'pytorch'
    device :
        Torch device required when model_type='pytorch'
    batch_size :
        Batch size for torch inference

    Returns
    -------
    np.ndarray
        Probabilities aligned to `class_order`, shape (N, n_classes)
    """
    if model_type == 'sklearn':
        probs = model.predict_proba(features)

    elif model_type == 'pytorch':
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.tensor(features[i:i + batch_size], dtype=torch.float32).to(device)
                logits = model(batch)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.vstack(probs_list)

    else:
        raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got {model_type}")

    return align_proba_to_class_order(probs, model_class_order, class_order)


def crop_img(img):
    """
    Crop image to the bounding box of the largest foreground object.
    Automatically detects the largest contiguous object in the image and crops
    to its bounding rectangle with a small padding.

    Parameters
    ----------
    img :
        Can be grayscale (H, W) or colored (H, W, C).

    Returns
    -------
        Cropped image containing the largest detected object with 5-pixel padding.
        Returns original image unchanged if:
        - No contours are detected
        - Cropped region would be smaller than 100x100 pixels
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:
        return img

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    add_pixels = 5
    height, width = img.shape[:2]

    x = max(0, x - add_pixels)
    y = max(0, y - add_pixels)
    w = min(width - x, w + 2 * add_pixels)
    h = min(height - y, h + 2 * add_pixels)

    new_img = img[y:y + h, x:x + w].copy()

    if new_img.shape[0] < 100 or new_img.shape[1] < 100:
        return img

    return new_img


class CroppedImage(Dataset):
    """
    PyTorch Dataset for loading images with optional automatic cropping.
    Uses OpenCV for image loading and preprocessing before applying transforms.
    """
    def __init__(self, root_dir, transform=None, apply_crop=True):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.apply_crop = apply_crop
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        img = cv2.imread(img_path)

        if self.apply_crop:
            try:
                img = crop_img(img)
            except Exception as e:
                print(f'Cropping failed for {img_path}: {e}')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def denorm_img(img_t, mean=0.5, std=0.5):
    """
        Denormalize a normalized image tensor for visualization.
        Converts a normalized PyTorch tensor (C, H, W) to a numpy array (H, W, C).

        Parameters
        ----------
        img_t : torch.Tensor
            Normalized image tensor of shape (C, H, W)
        mean : float, optional
            Mean value used during normalization
        std : float, optional
            Standard deviation used during normalization

        Returns
        -------
        numpy.ndarray
            Denormalized image array of shape (H, W, C) with values clipped to [0, 1]
        """
    img = img_t.detach().cpu().float()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()


def show_heatmap_per_class(x_images, importance_maps, labels, class_names, n_classes, alpha=0.45, cmap='jet',
                           save_path=None):
    """
    Display Grad-CAM heatmap overlays for one sample from each class.
    Creates a visualization showing the original image and Grad-CAM heatmap for the first sample of each class.

    Parameters
    ----------
    x_images : torch.Tensor
        Image tensors of shape (N, C, H, W)
    importance_maps : numpy.ndarray
        Importance maps of shape (N, H, W) with values in [0, 1]
    labels : numpy.ndarray
        Class labels with integer class indices
    class_names : list of str
        Names of classes in order corresponding to class indices
    n_classes : int
        Total number of classes
    alpha : float, optional
        Transparency of heatmap overlay in range [0, 1]
    cmap : str, optional
        Matplotlib colormap name for heatmap
    save_path : str, optional
        Path to save figure. If None, displays interactively

    Returns
    -------
    None
        Displays or saves the figure
    """
    fig, axes = plt.subplots(n_classes, 2, figsize=(10, 5 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_name in enumerate(class_names):
        # Find first image of this class
        idx = np.where(labels == class_idx)[0][0]

        # Get image and heatmap
        img = denorm_img(x_images[idx])
        hm = np.clip(importance_maps[idx], 0, 1)

        # Plot original image
        axes[class_idx, 0].imshow(img)
        axes[class_idx, 0].set_title(f'{class_name} - Original')
        axes[class_idx, 0].axis('off')

        # Plot heatmap overlay
        axes[class_idx, 1].imshow(img)
        axes[class_idx, 1].imshow(hm, alpha=alpha, cmap=cmap)
        axes[class_idx, 1].set_title(f'{class_name} - Grad-CAM')
        axes[class_idx, 1].axis('off')

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def show_occlusions_same_idx(x_images, patch_rankings, patch_meta, idx=0, fractions=(0.0, 0.2, 0.4, 0.6, 0.8, 1),
                             baseline='blur', blur_ksize=31, blur_sigma=7.0, n_cols=3, save_path=None):
    """
        Visualize progressive occlusion of image regions based on Grad-CAM importance.

        Shows how an image looks when increasingly important regions (ranked by
        Grad-CAM) are occluded using a specified baseline strategy. Helps
        validate that the importance maps correctly identify critical regions.

        Parameters
        ----------
        x_images : torch.Tensor
            Image tensors of shape (N, C, H, W)
        patch_rankings : list of numpy.ndarray
            List of N arrays, each containing patch indices sorted by importance (most important first)
        patch_meta : dict
            Dictionary containing patch metadata with keys:
            - 'patch_size': tuple of (height, width) for patches
            - Other metadata needed by apply_importance_masking
        idx : int, optional
            Index of image to visualize
        fractions : tuple of float, optional
            Fractions of image to occlude, values in [0, 1]
        baseline : str, optional
            Occlusion strategy
        blur_ksize : int, optional
            Kernel size for blur baseline, must be odd
        blur_sigma : float, optional
            Sigma parameter for Gaussian blur
        n_cols : int, optional
            Number of columns in subplot grid
        save_path : str, optional
            Path to save figure. If None, displays interactively

        Returns
        -------
        None
            Displays or saves the figure
        """

    img0 = x_images[idx:idx+1]

    n_rows = int(np.ceil(len(fractions) / n_cols))
    plt.figure(figsize=(4*n_cols, 4*n_rows))

    for j, frac in enumerate(fractions, 1):
        img_occ = apply_importance_masking(
            images=img0,
            patch_rankings=[patch_rankings[idx]],
            patch_meta=patch_meta,
            fraction_to_mask=frac,
            mask_strategy='most_important',
            baseline=baseline,
            blur_ksize=blur_ksize,
            blur_sigma=blur_sigma
        )[0]

        ax = plt.subplot(n_rows, n_cols, j)
        ax.imshow(denorm_img(img_occ))
        ax.set_title(f'{int(frac*100)}% occluded')
        ax.axis('off')

    plt.suptitle(
        'Grad-CAM–guided occlusion (blur baseline)',
        fontsize=14,
        y=1.02
    )
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def fill_nan_tail(vec):
    vec = np.asarray(vec, dtype=float).copy()
    bad = np.where(~np.isfinite(vec))[0]
    if len(bad) > 0:
        vec[bad[0]:] = 0.0
    return vec


def aurga_from_curve(curve):
    curve = fill_nan_tail(curve)
    x = np.linspace(0, 1, len(curve))
    return auc(x, curve)


def ideal_prob_matrix(y_labels, class_order):
    y_labels = np.asarray(y_labels)
    class_order = np.asarray(class_order)
    n = len(y_labels)
    ideal = np.zeros((n, len(class_order)), dtype=np.float32)
    for k, c in enumerate(class_order):
        ideal[:, k] = (y_labels == c).astype(np.float32)
    return ideal