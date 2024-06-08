
import torch
import torch.nn as nn
import torch.fft

class FSDLoss(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FSDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def loss_formulation(self, recon_freq, real_freq):
        """Calculate the loss function"""
        # Construct the spectrum weight matrix
        matrix_tmp = (recon_freq - real_freq) ** 2
        matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

        # Normalize the matrix
        max_val = torch.amax(matrix_tmp, dim=[-1, -2], keepdim=True)
        matrix_tmp = matrix_tmp / max_val
        matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
        matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
        weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of the spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # Calculate frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # Dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, **kwargs):
        """Forward function to calculate FSD loss

        Args:
            pred (torch.Tensor): Shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): Shape (N, C, H, W). Target tensor.
        """
        # Convert images to frequency domain representation
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        pred_freq = torch.stack([pred_freq.real, pred_freq.imag], -1)
        target_freq = torch.fft.fft2(target, norm='ortho')
        target_freq = torch.stack([target_freq.real, target_freq.imag], -1)

        return self.loss_formulation(pred_freq, target_freq) * self.loss_weight