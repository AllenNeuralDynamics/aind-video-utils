"""Tools for working with video files using ffmpeg."""

from importlib.metadata import PackageNotFoundError, version

from aind_video_utils._rawvideo import pix_format_bit_depth
from aind_video_utils.color_spaces import linear_to_rec_709_trc, luma_range, rec_709_trc_to_linear
from aind_video_utils.encoding import (
    OFFLINE_8BIT,
    OFFLINE_10BIT,
    ONLINE_8BIT,
    ONLINE_10BIT,
    SPEC_VERSION,
    EncodingProfile,
    RangeOverride,
    with_setparams,
)
from aind_video_utils.frames import extract_frame_by_index, extract_luma_frame, extract_srgb_frame
from aind_video_utils.mp4_index import EditListEntry, Mp4FrameIndex, read_mp4_frame_index
from aind_video_utils.probe import (
    get_color_transfer,
    get_frame_dimensions,
    get_nb_frames,
    get_r_frame_rate,
    get_video_range_info,
    probe,
)
from aind_video_utils.transcode import VIDEO_EXTENSIONS, transcode_video

try:
    __version__ = version("aind-video-utils")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # encoding profiles
    "EncodingProfile",
    "OFFLINE_8BIT",
    "OFFLINE_10BIT",
    "ONLINE_8BIT",
    "ONLINE_10BIT",
    "RangeOverride",
    "SPEC_VERSION",
    "with_setparams",
    # frames
    "extract_frame_by_index",
    "extract_luma_frame",
    "extract_srgb_frame",
    # mp4_index
    "EditListEntry",
    "Mp4FrameIndex",
    "read_mp4_frame_index",
    # probe
    "get_color_transfer",
    "get_frame_dimensions",
    "get_nb_frames",
    "get_r_frame_rate",
    "get_video_range_info",
    "probe",
    # _rawvideo
    "pix_format_bit_depth",
    # color_spaces
    "luma_range",
    "linear_to_rec_709_trc",
    "rec_709_trc_to_linear",
    # transcode
    "transcode_video",
    "VIDEO_EXTENSIONS",
]
