import torch

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):

        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)


        intersection = (prediction_flat * target_flat).sum()
        union = prediction_flat.sum() + target_flat.sum()


        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)


        dice_loss = 1 - dice_coefficient

        return dice_loss

class DiceWithLogitsLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceWithLogitsLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction=torch.sigmoid(prediction)

        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)


        intersection = (prediction_flat * target_flat).sum()
        union = prediction_flat.sum() + target_flat.sum()


        dice_coefficient = (2. * intersection + self.smooth) / (union + self.smooth)


        dice_loss = 1 - dice_coefficient

        return dice_loss