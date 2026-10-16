"""Where H3's video lives in a sampling-time latent: [video, audio], so the shared reader applies."""

from ..._core import streams as _streams

video_stream = _streams.av_video_stream
