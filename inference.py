from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import RigidTransform3D, TranslationTransform3D
from scipy.ndimage import zoom

from predict_decoders import LoberModule
from predict_lung import LungModule
from predict_normal import LoberModuleNormal
from utils.general import (
    calculate_similarity_metrics,
    find_best_registration,
    find_files,
    pos_processamento,
    post_processing_dist_lung,
    post_processing_lung,
)
from utils.to_onehot import mask_to_onehot


REPO_ROOT = Path(__file__).resolve().parent
RAW_DATA_FOLDER = REPO_ROOT / "raw_images"
WEIGHTS_DIR = REPO_ROOT / "weights"


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("LobePrior in-memory inference currently requires CUDA.")


def _required_paths(use_prior):
    required = [
        WEIGHTS_DIR / "LightningLung.ckpt",
        REPO_ROOT / "raw_images" / "images_npz",
    ]
    if use_prior:
        required.extend(
            [
                WEIGHTS_DIR / "LightningLobes.ckpt",
                RAW_DATA_FOLDER / "groups",
                RAW_DATA_FOLDER / "model_fusion",
            ]
        )
    else:
        required.append(WEIGHTS_DIR / "LightningLobes_no_template.ckpt")
    return required


def _assert_runtime_assets(use_prior):
    missing = [str(path) for path in _required_paths(use_prior) if not path.exists()]
    if missing:
        raise FileNotFoundError("LobePrior is missing required runtime files:\n- " + "\n- ".join(missing))


def _flip_axes_from_direction(sitk_image):
    directions = np.asarray(sitk_image.GetDirection())
    if len(directions) != 9:
        return np.array([], dtype=int)
    return np.where(directions[[0, 4, 8]][::-1] < 0)[0]


def _preprocess_image(sitk_image):
    original = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    flip_axes = _flip_axes_from_direction(sitk_image)
    canonical = np.flip(original, flip_axes).copy() if flip_axes.size else original.copy()
    spacing = sitk_image.GetSpacing()[::-1]
    isometric = zoom(canonical, spacing).astype(np.float32)
    isometric = np.clip(isometric, -1024.0, 600.0)
    isometric = (isometric + 1024.0) / 1624.0
    return canonical, isometric, flip_axes


def _resize_128(image_zyx):
    subject = tio.Subject(image=tio.ScalarImage(tensor=np.expand_dims(image_zyx, 0)))
    transformed = tio.Resize((128, 128, 128))(subject)
    return transformed.image.numpy()


def _build_normal_sample(image_zyx):
    image = np.expand_dims(image_zyx.astype(np.float32), 0)
    image_high = _resize_128(image_zyx)
    return {
        "image_h": torch.tensor(image_high, dtype=torch.float32).unsqueeze(0).cuda(),
        "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0).cuda(),
    }


def _build_prior_sample(registered_xyz, group):
    template_path = RAW_DATA_FOLDER / "model_fusion" / f"group_{group}.npz"
    template = np.load(template_path)["model"][:].astype(np.float32)
    image_zyx = registered_xyz.transpose(2, 1, 0).astype(np.float32)
    image = np.expand_dims(image_zyx, 0)
    image_high = _resize_128(image_zyx)
    return {
        "image_h": torch.tensor(image_high, dtype=torch.float32).unsqueeze(0).cuda(),
        "image": torch.tensor(image, dtype=torch.float32).unsqueeze(0).cuda(),
        "template": torch.tensor(template, dtype=torch.float32).unsqueeze(0).cuda(),
    }


def _resize_xyz_nearest(label_xyz, target_shape_xyz):
    tensor = torch.from_numpy(label_xyz.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(tensor, size=target_shape_xyz, mode="nearest")
    return resized.squeeze().numpy()


def _restore_to_original(label_zyx, rigid, canonical_shape_zyx, flip_axes):
    label_xyz = label_zyx.transpose(2, 1, 0).astype(np.float32)
    if rigid is not None:
        label_xyz = rigid.transform_inverse(label_xyz, interpolation="nearest")
    restored_xyz = _resize_xyz_nearest(label_xyz, canonical_shape_zyx[::-1])
    restored_zyx = restored_xyz.transpose(2, 1, 0)
    if flip_axes.size:
        restored_zyx = np.flip(restored_zyx, flip_axes).copy()
    return restored_zyx.astype(np.uint8)


def _register_group(image_xyz, group):
    fixed_path = RAW_DATA_FOLDER / "groups" / f"group_{group}.nii.gz"
    template_img = nib.load(str(fixed_path))
    template_data = template_img.get_fdata()
    template_affine = template_img.affine
    moving_affine = np.eye(4, dtype=np.float32)

    c_of_mass = transform_centers_of_mass(
        template_data, template_affine, image_xyz, moving_affine
    )

    metric = MutualInformationMetric(32, None)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=[10000, 1000, 100],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )

    translation = affreg.optimize(
        template_data,
        image_xyz,
        TranslationTransform3D(),
        None,
        template_affine,
        moving_affine,
        starting_affine=c_of_mass.affine,
    )
    rigid = affreg.optimize(
        template_data,
        image_xyz,
        RigidTransform3D(),
        None,
        template_affine,
        moving_affine,
        starting_affine=translation.affine,
    )
    transformed = rigid.transform(image_xyz)
    return transformed.astype(np.float32), rigid


def _pick_best_group(registered_by_group):
    results = {}
    for reg_file in sorted((RAW_DATA_FOLDER / "images_npz").glob("*.npz")):
        moving = np.load(reg_file)
        group = int(moving["group"])
        if group not in registered_by_group:
            continue
        moving_array = moving["image"][:].astype(np.float32).transpose(2, 1, 0)
        reference_array = registered_by_group[group]["image"].transpose(2, 1, 0)
        results[reg_file.name] = calculate_similarity_metrics(reference_array, moving_array)

    if not results:
        raise RuntimeError("No LobePrior registration templates were available to score.")

    best_image, _ = find_best_registration(results)
    best_npz = np.load(RAW_DATA_FOLDER / "images_npz" / best_image)
    return int(best_npz["group"])


def _postprocess_normal(output_lobes, lung):
    image = mask_to_onehot(output_lobes)
    image = np.expand_dims(image, 0)
    for channel in range(1, image.shape[1]):
        image[0, channel] = post_processing_lung(image[0, channel])
    image = torch.from_numpy(image)
    image = image.squeeze().argmax(dim=0).numpy().astype(np.int8)
    return post_processing_dist_lung(image, lung)


def _postprocess_prior(image, template, lung):
    lung_tensor = torch.from_numpy(lung).float()
    image = pos_processamento(
        output=image.cpu(),
        template=template.cpu(),
        segmentation=lung_tensor.unsqueeze(0).unsqueeze(0),
    )
    lung_uint8 = lung.astype(np.uint8)

    image = image.squeeze().numpy()
    image = mask_to_onehot(image)
    image = np.expand_dims(image, 0)
    for channel in range(1, image.shape[1]):
        image[0, channel] = post_processing_lung(image[0, channel])
    image = torch.from_numpy(image)
    image = image.squeeze().argmax(dim=0).numpy().astype(np.int8)
    return post_processing_dist_lung(image, lung_uint8)


@lru_cache(maxsize=1)
def _load_lung_model():
    _require_cuda()
    return LungModule.load_from_checkpoint(
        str(WEIGHTS_DIR / "LightningLung.ckpt"),
        strict=False,
        weights_only=False,
    )


@lru_cache(maxsize=1)
def _load_prior_model():
    _require_cuda()
    return LoberModule.load_from_checkpoint(
        str(WEIGHTS_DIR / "LightningLobes.ckpt"),
        strict=False,
        weights_only=False,
    )


@lru_cache(maxsize=1)
def _load_normal_model():
    _require_cuda()
    return LoberModuleNormal.load_from_checkpoint(
        str(WEIGHTS_DIR / "LightningLobes_no_template.ckpt"),
        strict=False,
        weights_only=False,
    )


def predict_lobes_in_memory(sitk_image, use_prior=True):
    _assert_runtime_assets(use_prior)
    canonical, isometric_zyx, flip_axes = _preprocess_image(sitk_image)

    lung_model = _load_lung_model()

    if use_prior:
        image_xyz = isometric_zyx.transpose(2, 1, 0).astype(np.float32)
        groups = sorted(set(find_files()))
        registered_by_group = {}
        for group in groups:
            registered_image, rigid = _register_group(image_xyz, group)
            registered_by_group[group] = {"image": registered_image, "rigid": rigid}

        best_group = _pick_best_group(registered_by_group)
        sample = _build_prior_sample(registered_by_group[best_group]["image"], best_group)

        lung = lung_model.predict(sample, "in_memory")

        lobe_model = _load_prior_model()
        lobe_model.eval()
        with torch.no_grad():
            _, image, _ = lobe_model.test_step(sample)

        labels = _postprocess_prior(image, sample["template"], lung)
        restored = _restore_to_original(
            labels,
            registered_by_group[best_group]["rigid"],
            canonical.shape,
            flip_axes,
        )
    else:
        sample = _build_normal_sample(isometric_zyx)
        lung = lung_model.predict(sample, "in_memory")

        lobe_model = _load_normal_model()
        lobe_model.eval()
        with torch.no_grad():
            output_lobes, _ = lobe_model.test_step(sample)

        labels = _postprocess_normal(output_lobes, lung)
        restored = _restore_to_original(labels, None, canonical.shape, flip_axes)

    sitk_lobes = sitk.GetImageFromArray(restored)
    sitk_lobes.CopyInformation(sitk_image)
    return sitk.Cast(sitk_lobes, sitk.sitkUInt8)
