import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class MultiHeadRetinaModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        vessel_classes: int = 1,
        centerline_classes: int = 1,
        av_classes: int = 3,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.vessel_classes = vessel_classes
        self.centerline_classes = centerline_classes
        self.av_classes = av_classes
        
        self.vessel_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=vessel_classes,
        )
        
        self.centerline_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=centerline_classes,
        )
        
        self.av_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=av_classes,
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        vessel_logits = self.vessel_model(x)
        centerline_logits = self.centerline_model(x)
        av_logits = self.av_model(x)
        return {
            'vessel': vessel_logits,
            'centerline': centerline_logits,
            'av': av_logits
        }


class CombinedLoss(nn.Module):
    def __init__(self, av_weight: float = 0.5):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.av_weight = av_weight
    
    def forward(
        self,
        outputs: dict,
        vessel_target: torch.Tensor,
        centerline_target: torch.Tensor,
        av_target: torch.Tensor,
        has_av_mask: torch.Tensor
    ) -> dict:
        vessel_bce = self.bce_loss(outputs['vessel'], vessel_target)
        vessel_dice = self.dice_loss(outputs['vessel'], vessel_target)
        vessel_loss = vessel_bce + vessel_dice
        
        centerline_bce = self.bce_loss(outputs['centerline'], centerline_target)
        centerline_dice = self.dice_loss(outputs['centerline'], centerline_target)
        centerline_loss = centerline_bce + centerline_dice
        
        if has_av_mask.any():
            av_logits_filtered = outputs['av'][has_av_mask]
            av_target_filtered = av_target[has_av_mask]
            if av_logits_filtered.numel() > 0:
                av_loss = self.ce_loss(av_logits_filtered, av_target_filtered)
            else:
                av_loss = torch.tensor(0.0, device=outputs['av'].device)
        else:
            av_loss = torch.tensor(0.0, device=outputs['av'].device)
        
        total_loss = vessel_loss + centerline_loss + self.av_weight * av_loss
        return {
            'total': total_loss,
            'vessel': vessel_loss,
            'centerline': centerline_loss,
            'av': av_loss
        }


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    device: str = "cuda"
) -> MultiHeadRetinaModel:
    model = MultiHeadRetinaModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )
    model = model.to(device)
    return model
