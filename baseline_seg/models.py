from __future__ import annotations

from typing import Any, Dict

from baseline_seg.prompting import get_prompt_channel_count


def _build_model_internal(cfg: Dict[str, Any], out_channels_override: int | None = None):
    try:
        from monai.networks.nets import SwinUNETR, UNETR, UNet
    except Exception as exc:
        raise RuntimeError("MONAI is required for model construction. Install monai.") from exc

    mcfg = cfg["model"]
    backbone = str(mcfg.get("backbone", "unet")).lower()

    in_channels = int(mcfg.get("in_channels", 1)) + int(get_prompt_channel_count(cfg))
    num_classes = int(out_channels_override if out_channels_override is not None else mcfg["num_classes"])
    roi_size = tuple(int(x) for x in mcfg.get("roi_size", [128, 128, 128]))

    if backbone == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=tuple(int(x) for x in mcfg.get("channels", [32, 64, 128, 256, 512])),
            strides=tuple(int(x) for x in mcfg.get("strides", [2, 2, 2, 2])),
            num_res_units=2,
            norm="instance",
        )

    if backbone == "unetr":
        return UNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            img_size=roi_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            res_block=True,
        )

    if backbone == "swinunetr":
        return SwinUNETR(
            img_size=roi_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=24,
            use_checkpoint=False,
        )

    raise ValueError(f"Unsupported backbone: {backbone}")


def build_model(cfg: Dict[str, Any]):
    """Construct a backbone model from config.

    Supported backbones: unet, unetr, swinunetr.
    """

    return _build_model_internal(cfg, out_channels_override=None)


def build_model_with_out_channels(cfg: Dict[str, Any], out_channels: int):
    """Build model with same backbone but custom output channels."""

    return _build_model_internal(cfg, out_channels_override=int(out_channels))
