"""DUC-02: quality-aware temporal attention pooling on top of the local A3 SkeletonModel.

w_t = softmax_t(a_t + log(epsilon + q'_t))  <=>  w_t ∝ exp(a_t) * (epsilon + q'_t)

q_t   per input frame = mean over 33 joints of feature channel 2 (MediaPipe visibility, already 0 on frames
      without a detected pose and on joints removed by augmentation), read from the raw input before data_bn.
q'_t  per attention step = mean of q over the input frames centred on the strided temporal conv output
      (kernel 2*stride-1, padding stride-1, padding excluded); stride 2 maps 32 frames -> 16 steps.
All-missing clips give q' = 0 everywhere, the prior is constant and the weights equal the original attention.
The prior adds no parameters; with quality_prior=False the forward pass is identical to SkeletonModel.
"""
import torch
import torch.nn.functional as F

from training.component_ablation.models import SkeletonModel

QUALITY_CHANNEL = 2
EPSILON = 0.1


def frame_quality(values):
    return values[:, QUALITY_CHANNEL].clamp(0, 1).mean(dim=2)


def temporal_quality(quality, steps):
    frames = quality.shape[1]
    if frames % steps:
        raise ValueError(f"Cannot map {frames} frames to {steps} attention steps")
    stride = frames // steps
    if stride == 1:
        return quality
    return F.avg_pool1d(quality[:, None], 2 * stride - 1, stride, stride - 1, count_include_pad=False)[:, 0]


class QualityAwareSkeletonModel(SkeletonModel):
    def __init__(self, channels=8, attention=True, dropout=0.3, quality_prior=True, epsilon=EPSILON):
        if not attention or channels <= QUALITY_CHANNEL:
            raise ValueError("Quality prior reweights attention and needs the visibility channel")
        if not 0 < epsilon:
            raise ValueError("epsilon must be positive")
        super().__init__(channels, attention, dropout)
        self.quality_prior, self.epsilon = quality_prior, float(epsilon)

    def forward(self, values):
        quality = frame_quality(values)
        batch, channels, frames, nodes = values.shape
        values = values.permute(0, 3, 1, 2).contiguous().view(batch, nodes * channels, frames)
        values = self.data_bn(values).view(batch, nodes, channels, frames).permute(0, 2, 3, 1)
        temporal = self.blocks(values).mean(dim=3)
        scores = self.temporal_attention(temporal)
        if self.quality_prior:
            scores = scores + torch.log(self.epsilon + temporal_quality(quality, temporal.shape[2]))[:, None]
        weights = torch.softmax(scores, dim=2)
        return self.classifier((temporal * weights).sum(dim=2)), weights.squeeze(1)
