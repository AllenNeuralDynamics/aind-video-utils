"""Tools for working with video files using ffmpeg."""

from importlib.metadata import PackageNotFoundError, version

from aind_video_utils._rawvideo import pix_format_bit_depth
from aind_video_utils.color_spaces import linear_to_rec_709_trc, luma_range, rec_709_trc_to_linear
from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    OFFLINE_10BIT,
    ONLINE_8BIT,
    ONLINE_10BIT,
    POSTER_FILENAME,
    PREVIEW_FILENAME,
    SPEC_VERSION,
    Derivative,
    EncodingProfile,
    RangeOverride,
    preview_decimation,
    with_poster,
    with_preview,
    with_setparams,
)
from aind_video_utils.frames import extract_frame_by_index, extract_luma_frame, extract_srgb_frame
from aind_video_utils.mp4_index import EditListEntry, Mp4FrameIndex, read_mp4_frame_index
from aind_video_utils.preview_metadata import PREVIEW_METADATA_FILENAME, write_preview_metadata
from aind_video_utils.probe import (
    get_color_transfer,
    get_exact_nb_frames,
    get_frame_dimensions,
    get_nb_frames,
    get_r_frame_rate,
    get_video_range_info,
    probe,
)
from aind_video_utils.transcode import VIDEO_EXTENSIONS, FfmpegError, transcode_video

try:
    __version__ = version("aind-video-utils")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "OFFLINE_8BIT",
    "OFFLINE_10BIT",
    "ONLINE_8BIT",
    "ONLINE_10BIT",
    "POSTER_FILENAME",
    "PREVIEW_FILENAME",
    "PREVIEW_METADATA_FILENAME",
    "SPEC_VERSION",
    "VIDEO_EXTENSIONS",
    "Derivative",
    "EditListEntry",
    "EncodingProfile",
    "FfmpegError",
    "Mp4FrameIndex",
    "RangeOverride",
    "__version__",
    "extract_frame_by_index",
    "extract_luma_frame",
    "extract_srgb_frame",
    "get_color_transfer",
    "get_exact_nb_frames",
    "get_frame_dimensions",
    "get_nb_frames",
    "get_r_frame_rate",
    "get_video_range_info",
    "linear_to_rec_709_trc",
    "luma_range",
    "pix_format_bit_depth",
    "preview_decimation",
    "probe",
    "read_mp4_frame_index",
    "rec_709_trc_to_linear",
    "transcode_video",
    "with_poster",
    "with_preview",
    "with_setparams",
    "write_preview_metadata",
]
