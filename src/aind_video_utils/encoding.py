"""Encoding profiles for the AIND behavior video file standard.

Defines frozen dataclass profiles that map 1-to-1 with the encoding profiles
in the `aind-file-standards behavior video spec
<https://allenneuraldynamics.github.io/aind-file-standards/file_formats/behavior_videos/>`_.

Each profile bundles every ffmpeg flag needed to produce a compliant file.
Select a preset constant and optionally customise it with
:meth:`EncodingProfile.replace`::

    from aind_video_utils.encoding import OFFLINE_8BIT
    fast = OFFLINE_8BIT.replace(codec_params=("-preset", "veryfast", "-crf", "18"))
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aind_video_utils.probe import (
    ProbeDict,
    get_color_primaries,
    get_color_range,
    get_color_space,
    get_color_transfer,
    get_nb_frames,
    get_r_frame_rate,
    get_yuv_format,
)

RangeOverride = Literal["pc", "tv"]

SPEC_VERSION: str = "0.3.0"
"""Version of the aind-file-standards behavior video spec these profiles
implement, matching the ``## Version`` heading of that document.  Independent
of this package's own version."""

# ---------------------------------------------------------------------------
# Setparams filter — fill missing color metadata only
# ---------------------------------------------------------------------------
#
# AIND sources carry partial or no color metadata in the bitstream:
#   - h264 gbrp files (camera-linear RGB) tag color_range=pc and color_space=gbr;
#     color_trc and color_primaries are absent.
#   - mpeg4 yuv420p files carry NO color metadata at all (the bitstream has no
#     VUI for it); the underlying RGB→YUV matrix is whatever libswscale used
#     when Bonsai/ffmpeg created the file, which empirically is smpte170m
#     (BT.601) — the universal default for untagged conversion.
#
# Defensive defaults injected here (only for fields the source doesn't tag):
#   color_primaries=bt709    — Modern camera sensors, and ffmpeg doesn't apply
#                              a primary rotation during RGB→YUV. Declaring
#                              bt709 means "no primary rotation" downstream.
#   color_trc=linear         — Bonsai stores raw linear scene light (no OETF
#                              applied at capture time).
#   colorspace=gbr / smpte170m — Match the bitstream truth: gbr for RGB-planar
#                              sources; smpte170m (BT.601) for untagged YUV,
#                              the matrix libswscale uses by default.
#   range=pc                 — Bonsai stores full-range Y (luma reaches 0 and
#                              255 in production files, incompatible with
#                              tv-range floor at 16). Without this, untagged
#                              yuv420p defaults to tv-range and crushes
#                              shadows/highlights at the chain's first scale.


def setparams_filter_for_source(
    probe_json: ProbeDict,
    range_override: RangeOverride | None = None,
) -> str | None:
    """Build a ``setparams`` filter string with only the fields missing on the source.

    Returns ``None`` if the source has color_primaries, color_trc, color_space,
    and color_range all tagged and no ``range_override`` is requested.  Otherwise
    returns ``setparams=<a=b:c=d:...>`` for use as a leading filter in the chain.

    Defaults follow the AIND Bonsai capture conventions documented above.

    Parameters
    ----------
    probe_json : ProbeDict
        Probe result for the source.
    range_override : {"pc", "tv"} | None
        When set, force the ``range=`` field to this value regardless of what
        the source tags.  Use ``"tv"`` for sources known to be TV-range encoded
        but tagged otherwise (e.g. the AIND mpeg4 yuv420p subset), which the
        default ``range=pc`` fallback would mis-tag.
    """
    parts: list[str] = []
    if get_color_primaries(probe_json) is None:
        parts.append("color_primaries=bt709")
    if get_color_transfer(probe_json) is None:
        parts.append("color_trc=linear")
    if get_color_space(probe_json) is None:
        pix_fmt = get_yuv_format(probe_json)
        if pix_fmt and pix_fmt.startswith("gbr"):
            parts.append("colorspace=gbr")
        else:
            parts.append("colorspace=smpte170m")
    if range_override is not None:
        parts.append(f"range={range_override}")
    elif get_color_range(probe_json) is None:
        parts.append("range=pc")
    if not parts:
        return None
    return f"setparams={':'.join(parts)}"


def _join_chains(*parts: str) -> str:
    """Join filter-chain fragments with commas, skipping empty ones."""
    return ",".join(part for part in parts if part)


def _codec_args(
    *,
    codec: str,
    pixel_format: str,
    codec_params: tuple[str, ...],
    metadata: tuple[tuple[str, str], ...],
    output_flags: tuple[str, ...],
) -> list[str]:
    """Build the per-output arguments that follow the filter graph.

    Order: ``-c:v``, ``-pix_fmt``, codec params, metadata, output flags.
    Shared by the primary output and every :class:`Derivative`.
    """
    args: list[str] = ["-c:v", codec, "-pix_fmt", pixel_format]
    args.extend(codec_params)
    for key, value in metadata:
        args.extend(["-metadata", f"{key}={value}"])
    args.extend(output_flags)
    return args


@dataclass(frozen=True)
class Derivative:
    """A secondary output encoded from a branch of a profile's shared chain.

    Derivatives turn a profile into a multi-output ffmpeg invocation: the chain
    in :attr:`EncodingProfile.video_filters` runs once, feeds a ``split``, and
    each branch reaches its own encoder.  A derivative therefore costs one
    extra encode, not an extra decode or an extra pass over the source.

    Parameters
    ----------
    suffix : str
        Appended to the primary output's stem, so ``"_preview"`` turns
        ``clip.mp4`` into ``clip_preview.mp4``.
    codec : str
        Value for ``-c:v``.
    pixel_format : str
        Value for ``-pix_fmt``.
    container : str
        File extension without dot.
    filters : str
        Tail filter chain applied after the split -- not a whole chain, since
        the shared colour and CFR processing has already run.  Commas inside a
        filter's arguments need backslash-escaping (``select=not(mod(n\\,20))``)
        because the graph parser reads a bare comma as a chain separator.
        Empty encodes the split output unchanged.
    codec_params : tuple[str, ...]
        Rate-control, tuning, preset flags.
    output_flags : tuple[str, ...]
        Container/muxer flags placed after codec options.
    metadata : tuple[tuple[str, str], ...]
        ``-metadata key=value`` pairs to embed.
    fps_mode : {"passthrough", "cfr", "vfr"}
        Value for ``-fps_mode``.  Leave it at ``"passthrough"`` for any
        frame-dropping derivative: ``select`` and friends drop frames without
        updating the filter link's advertised frame rate, so ffmpeg's default
        CFR stage duplicates every retained frame back up to the source rate.
        ``"passthrough"`` also cannot duplicate frames at all, which is what
        keeps a nonzero process-wide ``dup_frames`` attributable to the primary
        output -- see ``transcode_video``'s ``fail_on_frame_drop``.
    tap : {"shared", "source"}
        Where this derivative branches from.  ``"shared"`` (the default) takes
        the output of :attr:`EncodingProfile.video_filters`, so the derivative
        encodes exactly the pixels the primary output does.  ``"source"``
        branches off the conditioned source instead -- after
        :attr:`EncodingProfile.source_filters` has repaired the source metadata,
        before any encoding -- and ``filters`` must then carry the whole
        encoding chain the derivative needs.

        A source tap therefore inherits the probe-derived ``setparams``, its
        ``range_override`` included, rather than restating it; conditioning is
        metadata-only, so the pixels it sees are still the source's own.

        Tap the source when a derivative needs a different transfer function
        than the archive.  A JPEG still is the motivating case: JFIF is read as
        sRGB, and converting the archive's BT.709 output to sRGB afterwards
        measures *worse* than leaving it alone, because zimg treats BT.709 as
        BT.1886 and the round trip compounds the error.  Branching ahead of the
        BT.709 OETF and encoding sRGB straight from linear light is accurate to
        within one code.

        Order ``select`` first in a source-tapped chain when only a few frames
        are wanted -- the rest of the chain then runs on those frames alone, so
        re-deriving the shared work costs almost nothing.
    """

    suffix: str
    codec: str
    pixel_format: str
    container: str
    filters: str = ""
    codec_params: tuple[str, ...] = ()
    output_flags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    fps_mode: Literal["passthrough", "cfr", "vfr"] = "passthrough"
    tap: Literal["shared", "source"] = "shared"

    def output_path_for(self, primary: Path) -> Path:
        """Return this derivative's path alongside the *primary* output."""
        return primary.with_name(f"{primary.stem}{self.suffix}.{self.container}")

    def ffmpeg_codec_args(self) -> list[str]:
        """Return this derivative's arguments between its ``-map`` and its path."""
        return [
            "-fps_mode",
            self.fps_mode,
            *_codec_args(
                codec=self.codec,
                pixel_format=self.pixel_format,
                codec_params=self.codec_params,
                metadata=self.metadata,
                output_flags=self.output_flags,
            ),
        ]


@dataclass(frozen=True)
class EncodingProfile:
    """Immutable bundle of ffmpeg parameters for a single encoding profile.

    Parameters
    ----------
    video_filters : str
        The encoding chain: whatever turns conditioned source pixels into the
        primary output's pixels.  Shared by every derivative that taps
        ``"shared"``.
    codec : str
        Value for ``-c:v``.
    pixel_format : str
        Value for ``-pix_fmt``.
    container : str
        File extension without dot (``"mp4"``, ``"mkv"``).
    codec_params : tuple[str, ...]
        Rate-control, tuning, preset flags (e.g. ``("-crf", "18")``).
    input_flags : tuple[str, ...]
        Flags placed before ``-i``.
    output_flags : tuple[str, ...]
        Container/muxer flags placed after codec options.
    metadata : tuple[tuple[str, str], ...]
        ``-metadata key=value`` pairs to embed.
    source_filters : str
        Source conditioning: filters that repair what the source's own metadata
        gets wrong, rather than encoding choices.  ``setparams`` (see
        :func:`with_setparams`) and the CFR-normalizing ``setpts`` belong here.
        These run ahead of every split, so each branch of a multi-output graph
        reads the source under one repaired interpretation instead of its own
        copy of one.  Conditioning is metadata-only and changes no pixel values.
    derivatives : tuple[Derivative, ...]
        Extra outputs branched off the shared chain.  Empty (the default)
        yields a single-output command identical to one built without
        multi-output support.
    """

    video_filters: str
    codec: str
    pixel_format: str
    container: str
    source_filters: str = ""
    codec_params: tuple[str, ...] = ()
    input_flags: tuple[str, ...] = ()
    output_flags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    derivatives: tuple[Derivative, ...] = ()

    def replace(self, **kwargs: Any) -> EncodingProfile:
        """Return a copy with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)

    def ffmpeg_input_args(self) -> list[str]:
        """Return the argument list to insert before ``-i``."""
        return list(self.input_flags)

    def prepend_conditioning(self, clause: str) -> EncodingProfile:
        """Return a copy with *clause* at the head of ``source_filters``.

        Use this for anything that repairs the source's own metadata, so it
        runs ahead of every split and all branches share one interpretation.
        """
        return self.replace(source_filters=_join_chains(clause, self.source_filters))

    def ffmpeg_graph_args(self) -> list[str]:
        """Return the filter-graph arguments.

        A profile with no derivatives emits ``["-vf", <conditioning>,<chain>]``.
        With derivatives the graph splits at up to two points: once after the
        source conditioning, for derivatives that tap ``"source"``, and once
        after the shared chain, for those that tap ``"shared"``::

            [0:v]<conditioning>,split=2[chain][d1];
              [chain]<shared>,split=2[main][d0];
              [d0]<tail>[d0out];[d1]<own chain>[d1out]

        Conditioning always precedes every split, so each branch reads the
        source under the same repaired metadata.  A derivative with empty
        ``filters`` contributes no chain segment and is mapped straight off its
        split output.
        """
        if not self.derivatives:
            return ["-vf", _join_chains(self.source_filters, self.video_filters)]
        from_source = [i for i, d in enumerate(self.derivatives) if d.tap == "source"]
        from_shared = [i for i, d in enumerate(self.derivatives) if d.tap == "shared"]
        segments: list[str] = []
        if from_source:
            head = f"[0:v]{self.source_filters}," if self.source_filters else "[0:v]"
            segments.append(f"{head}split={len(from_source) + 1}[chain]" + "".join(f"[d{i}]" for i in from_source))
            chain = f"[chain]{self.video_filters}"
        else:
            chain = f"[0:v]{_join_chains(self.source_filters, self.video_filters)}"
        if from_shared:
            chain += f",split={len(from_shared) + 1}[main]" + "".join(f"[d{i}]" for i in from_shared)
        else:
            chain += "[main]"
        segments.append(chain)
        for i, derivative in enumerate(self.derivatives):
            if derivative.filters:
                segments.append(f"[d{i}]{derivative.filters}[d{i}out]")
        return ["-filter_complex", ";".join(segments)]

    def ffmpeg_output_groups(self) -> list[list[str]]:
        """Return one argument group per output, primary first, paths excluded.

        Each group covers everything between the filter graph and that output's
        path.  A ``-map`` leads each group only when the profile has
        derivatives, so a plain profile still emits exactly the arguments it
        did before multi-output support existed.
        """
        primary = _codec_args(
            codec=self.codec,
            pixel_format=self.pixel_format,
            codec_params=self.codec_params,
            metadata=self.metadata,
            output_flags=self.output_flags,
        )
        if not self.derivatives:
            return [primary]
        groups = [["-map", "[main]", *primary]]
        for i, derivative in enumerate(self.derivatives):
            label = f"[d{i}out]" if derivative.filters else f"[d{i}]"
            groups.append(["-map", label, *derivative.ffmpeg_codec_args()])
        return groups

    def output_paths(self, primary: Path) -> list[Path]:
        """Return every path this profile writes, primary first."""
        return [primary, *(d.output_path_for(primary) for d in self.derivatives)]

    def ffmpeg_output_args(self) -> list[str]:
        """Return the argument list to insert after ``-i <input>``.

        Order: ``-vf``, ``-c:v``, ``-pix_fmt``, codec params, metadata,
        output flags.

        Raises
        ------
        ValueError
            If the profile carries derivatives, which need one argument group
            and one path per output.  Returning just the primary leg here would
            silently drop the derivative outputs, so build the command from
            :meth:`ffmpeg_graph_args`, :meth:`ffmpeg_output_groups` and
            :meth:`output_paths` instead.
        """
        if self.derivatives:
            raise ValueError(
                f"profile has {len(self.derivatives)} derivative output(s); build the command from "
                "ffmpeg_graph_args(), ffmpeg_output_groups() and output_paths() instead."
            )
        return [*self.ffmpeg_graph_args(), *self.ffmpeg_output_groups()[0]]


# ---------------------------------------------------------------------------
# Canonical profiles — directly from the spec
# ---------------------------------------------------------------------------

_AIND_METADATA: tuple[tuple[str, str], ...] = (("author", "Allen Institute for Neural Dynamics"),)

OFFLINE_8BIT = EncodingProfile(
    video_filters=(
        "scale=out_color_matrix=bt709:out_range=full"
        ":flags=accurate_rnd+full_chroma_int+full_chroma_inp:sws_dither=none,"
        "format=yuv420p10le,"
        "colorspace=all=bt709:dither=none,"
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=ed,"
        "format=yuv420p"
    ),
    codec="libx264",
    pixel_format="yuv420p",
    container="mp4",
    codec_params=("-preset", "slow", "-crf", "18"),
    input_flags=(),
    output_flags=("-movflags", "+faststart+write_colr"),
    metadata=_AIND_METADATA,
)

OFFLINE_10BIT = EncodingProfile(
    video_filters=(
        "colorspace=all=bt709:dither=none,"
        "scale=out_range=tv:flags=accurate_rnd+full_chroma_int:sws_dither=none,"
        "format=yuv420p10le"
    ),
    codec="libx264",
    pixel_format="yuv420p10le",
    container="mp4",
    codec_params=("-preset", "slow", "-crf", "18"),
    input_flags=(),
    output_flags=("-movflags", "+faststart+write_colr"),
    metadata=_AIND_METADATA,
)

ONLINE_8BIT = EncodingProfile(
    video_filters=("scale=out_range=full,setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear"),
    codec="h264_nvenc",
    pixel_format="yuv420p",
    container="mkv",
    codec_params=("-tune", "hq", "-preset", "p3", "-rc", "vbr", "-cq", "18", "-b:v", "0M"),
    input_flags=(
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_range",
        "full",
        "-color_trc",
        "linear",
    ),
    output_flags=(
        "-color_range",
        "full",
        "-colorspace",
        "bt709",
        "-color_trc",
        "linear",
        "-maxrate",
        "700M",
        "-bufsize",
        "350M",
        "-f",
        "matroska",
        "-write_crc32",
        "0",
    ),
    metadata=_AIND_METADATA,
)

ONLINE_10BIT = EncodingProfile(
    video_filters=(
        "format=yuv420p10le,"
        "scale=out_range=full,"
        "setparams=range=full:colorspace=bt709:color_primaries=bt709:color_trc=linear"
    ),
    codec="hevc_nvenc",
    pixel_format="p010le",
    container="mkv",
    codec_params=("-tune", "hq", "-preset", "p4", "-rc", "vbr", "-cq", "12", "-b:v", "0M"),
    input_flags=(),
    output_flags=(
        "-color_range",
        "full",
        "-colorspace",
        "bt709",
        "-color_trc",
        "linear",
        "-maxrate",
        "700M",
        "-bufsize",
        "350M",
        "-f",
        "matroska",
        "-write_crc32",
        "0",
    ),
    metadata=_AIND_METADATA,
)

# ---------------------------------------------------------------------------
# Profile lookup by name (used by CLI)
# ---------------------------------------------------------------------------

PROFILES: dict[str, EncodingProfile] = {
    "offline-8bit": OFFLINE_8BIT,
    "offline-10bit": OFFLINE_10BIT,
    "online-8bit": ONLINE_8BIT,
    "online-10bit": ONLINE_10BIT,
}


def with_setparams(
    profile: EncodingProfile,
    probe_json: ProbeDict | None = None,
    range_override: RangeOverride | None = None,
) -> EncodingProfile:
    """Add a ``setparams`` colour-metadata filter to *profile*'s conditioning.

    Parameters
    ----------
    profile : EncodingProfile
        Base profile to extend.
    probe_json : ProbeDict | None
        Result of :func:`aind_video_utils.probe.probe` on the source.  When
        provided, the prepended setparams clause includes only fields the
        source has tagged as missing (per :func:`setparams_filter_for_source`),
        respecting any color metadata the source already declares.  When
        ``None``, every field is set to the AIND default — useful when the
        caller knows the source is fully untagged or doesn't want to probe.
    range_override : {"pc", "tv"} | None
        When set, force the ``range=`` field of the prepended setparams clause
        to this value.  Use ``"tv"`` for sources known to be TV-range encoded
        but tagged otherwise (e.g. the AIND mpeg4 yuv420p subset).  Defaults to
        ``None``, which falls back to the source tag (or ``"pc"`` if untagged).

    Returns
    -------
    EncodingProfile
        A new profile with the setparams clause added to ``source_filters``,
        or the original profile unchanged when ``probe_json`` indicates the
        source already carries all four color fields and no ``range_override``
        is requested.  Landing in ``source_filters`` rather than at the head of
        the encoding chain is what lets a source-tapped derivative read the
        source under this same clause instead of a hand-written copy of it.
    """
    if probe_json is not None:
        sp = setparams_filter_for_source(probe_json, range_override=range_override)
        if sp is None:
            return profile
    else:
        # No probe — default to filling every field. ``colorspace=smpte170m``
        # matches the bitstream truth for the typical untagged-yuv420p caller;
        # for gbrp callers without a probe, ``setparams`` is harmless metadata
        # (the scale step does RGB→YUV explicitly via ``out_color_matrix=``).
        range_value = range_override or "pc"
        sp = f"setparams=color_primaries=bt709:color_trc=linear:colorspace=smpte170m:range={range_value}"
    return profile.replace(source_filters=_join_chains(profile.source_filters, sp))


def preview_decimation(
    probe_json: ProbeDict,
    *,
    target_fps: float = 30.0,
    fps_band: tuple[float, float] = (25.0, 35.0),
) -> tuple[int, Fraction]:
    """Choose the integer decimation factor for a preview of this source.

    Keeping every *N*-th frame is what makes the preview deterministic -- every
    preview frame is a real source frame, and preview frame *k* is source frame
    *kN* -- but it also means the only achievable preview rates are ``S / N``
    for integer *N*.  Selection is therefore a choice of *N*, made in two steps.

    ``fps_band`` is a hard constraint: only factors whose resulting rate falls
    inside it are considered.  Within that set a rate that comes out an exact
    integer wins, and ties break toward ``target_fps``.  The band is what makes
    the integer preference safe rather than merely tidy: unbounded, a source
    rate with no useful divisors degenerates badly, and a 499 fps source would
    decimate to 1 fps because that is its nearest integer-valued rate.

    A band narrow enough to admit no factor at all falls back to the nearest
    rate to ``target_fps``, out of band, rather than refusing to build a
    preview; the returned rate says what actually happened.  Sources already
    slower than the band land on ``N = 1`` by the same route, since dropping
    frames cannot speed a video up.

    Parameters
    ----------
    probe_json : ProbeDict
        Probe result for the source, read for ``r_frame_rate``.
    target_fps : float
        Preferred rate within the band, used to rank admissible factors.
    fps_band : tuple[float, float]
        Inclusive ``(low, high)`` bounds on the preview rate.

    Returns
    -------
    tuple[int, Fraction]
        The decimation factor and the exact preview frame rate it produces.

    Raises
    ------
    RuntimeError
        If the source has no readable ``r_frame_rate``.
    ValueError
        If ``target_fps`` is not positive, the band is not a positive ordered
        pair, or ``target_fps`` falls outside the band.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    low, high = fps_band
    if not 0 < low <= high:
        raise ValueError(f"fps_band must be a positive (low, high) pair, got {fps_band}")
    if not low <= target_fps <= high:
        raise ValueError(f"target_fps {target_fps} falls outside fps_band {fps_band}")
    rate = get_r_frame_rate(probe_json)
    if rate is None:
        raise RuntimeError(
            "preview_decimation needs a readable r_frame_rate to size the decimation factor, "
            "and the source reports none."
        )
    source = Fraction(*rate)
    # Scan one factor wider than the band implies, then filter exactly, so
    # float bounds cannot round a boundary factor out of the candidate set.
    first = max(1, math.floor(source / high))
    last = max(1, math.ceil(source / low))
    admissible = [n for n in range(first, last + 1) if low <= source / n <= high]
    if not admissible:
        factor = max(1, round(source / target_fps))
    else:
        whole = [n for n in admissible if (source / n).denominator == 1]
        factor = min(whole or admissible, key=lambda n: (abs(float(source) / n - target_fps), n))
    return factor, source / factor


def with_preview(
    profile: EncodingProfile,
    probe_json: ProbeDict,
    *,
    target_fps: float = 30.0,
    fps_band: tuple[float, float] = (25.0, 35.0),
    suffix: str = "_preview",
    crf: int = 24,
    x264_preset: str = "veryfast",
) -> EncodingProfile:
    """Append a frame-decimated, browser-playable preview derivative to *profile*.

    The preview exists so a reviewer can stream a behavior video in a browser
    instead of downloading it.  Frame rate, not resolution, is what makes the
    archival file unplayable: AIND behavior sources run at 120-500 fps, which no
    browser will render in real time, while the frame is already small enough
    that scaling buys little.  The preview therefore keeps the source geometry
    and drops frames.

    Decimation keeps every *N*-th frame via ``select``, with *N* chosen by
    :func:`preview_decimation` from ``target_fps`` and ``fps_band``.  A
    resampling filter such as ``fps`` would instead synthesise a cadence
    whenever the ratio is not an integer, costing the property that makes this
    useful for QC: preview frame *k* is source frame *kN*, exactly.

    No ``setpts`` accompanies the ``select``: ``transcode_video``'s
    ``normalize_cfr`` has already rebased the shared chain to PTS 0, and frame 0
    always survives ``mod(n, factor)``, so the retained frames keep both their
    zero origin and their real-time spacing.

    The preview ends at its last retained frame, so it can end up to ``N - 1``
    source frames before the archive (38 ms for 500 fps decimated by 20).

    Parameters
    ----------
    profile : EncodingProfile
        Base profile; its ``metadata`` is copied onto the preview.
    probe_json : ProbeDict
        Probe result for the source, read for ``r_frame_rate``.
    target_fps : float
        Preferred preview rate, used to rank the factors the band admits.
    fps_band : tuple[float, float]
        Inclusive bounds on the preview rate.  See :func:`preview_decimation`
        for how the two interact and why the band is a hard constraint.
    suffix : str
        Stem suffix for the preview file.
    crf : int
        x264 quality for the preview.  Decimation dominates the size and
        streamability, so this is not a sensitive knob.
    x264_preset : str
        x264 speed preset.  Kept fast because two encoders share the machine
        with the archival encode, whose throughput is what actually matters.

    Returns
    -------
    EncodingProfile
        A copy of *profile* with the preview appended to ``derivatives``.

    Raises
    ------
    RuntimeError
        If the source has no readable ``r_frame_rate`` to size the factor from.
    ValueError
        If *profile* already has a derivative using ``suffix`` (both would write
        the same path), or if ``target_fps`` and ``fps_band`` are inconsistent.
    """
    if any(d.suffix == suffix for d in profile.derivatives):
        raise ValueError(f"profile already has a derivative with suffix {suffix!r}; both would write the same path.")
    factor, preview_fps = preview_decimation(probe_json, target_fps=target_fps, fps_band=fps_band)
    # factor == 1 means the source is already at or below the band; an empty
    # chain is clearer than a select that keeps every frame.
    filters = f"select=not(mod(n\\,{factor}))" if factor > 1 else ""
    derivative = Derivative(
        suffix=suffix,
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        filters=filters,
        # Two seconds per GOP: x264's 250-frame default is a 10 s keyframe
        # interval at preview rates, which makes browser scrubbing sluggish.
        codec_params=("-preset", x264_preset, "-crf", str(crf), "-g", str(max(1, round(2 * preview_fps)))),
        # write_colr matters as much here as on the archive: an untagged preview
        # renders with different colour than the file it stands in for.
        output_flags=("-movflags", "+faststart+write_colr"),
        metadata=profile.metadata,
    )
    return profile.replace(derivatives=(*profile.derivatives, derivative))


def with_poster(
    profile: EncodingProfile,
    probe_json: ProbeDict,
    *,
    at_seconds: float = 1.0,
    suffix: str = "_poster",
    quality: int = 3,
) -> EncodingProfile:
    """Append a JPEG still branched off the conditioned source.

    A poster gives QC pages, dashboards and ``<video poster=...>`` a cheap
    thumbnail without every viewer decoding the video to show one image.

    The still taps the source rather than the shared chain because JFIF carries
    no colour tags and every viewer reads a JPEG as sRGB, while the archive is
    BT.709.  Converting the archive's output to sRGB afterwards measures worse
    than not converting at all -- zimg treats BT.709 as BT.1886, so the round
    trip compounds the error rather than removing it.  Encoding sRGB straight
    from linear light lands within one code instead.

    Tapping the source costs nothing extra: ``select`` runs first, so the rest
    of the chain sees one frame.  It also means the still inherits the profile's
    ``source_filters`` -- the probe-derived ``setparams``, ``range_override``
    included -- so it reads the source exactly as the archive does.  Without
    that conditioning ``zscale`` has no transfer to resolve and the encode fails
    outright, which is why this needs no setparams clause of its own.

    Parameters
    ----------
    profile : EncodingProfile
        Base profile.
    probe_json : ProbeDict
        Probe result for the source, read for ``r_frame_rate`` and the frame
        count.
    at_seconds : float
        How far into the video to sample.  Sampling a little way in beats frame
        0, which is often blank, dark or mid-transition.  The frame index is
        clamped to the last frame when the probe reports a count.
    suffix : str
        Stem suffix for the still.
    quality : int
        Value for ``-q:v``; 2 is best, 31 worst.

    Returns
    -------
    EncodingProfile
        A copy of *profile* with the still appended to ``derivatives``.

    Raises
    ------
    RuntimeError
        If the source has no readable ``r_frame_rate``.
    ValueError
        If ``at_seconds`` is negative, or *profile* already has a derivative
        using ``suffix``.

    Notes
    -----
    When the probe reports no frame count and the video is shorter than
    ``at_seconds``, ``select`` matches nothing: ffmpeg writes no still and still
    exits zero, so the absence is silent.

    The still carries no ``-metadata``; ffmpeg's image2 muxer does not reliably
    embed it in a JPEG.
    """
    if at_seconds < 0:
        raise ValueError(f"at_seconds must not be negative, got {at_seconds}")
    if any(d.suffix == suffix for d in profile.derivatives):
        raise ValueError(f"profile already has a derivative with suffix {suffix!r}; both would write the same path.")
    rate = get_r_frame_rate(probe_json)
    if rate is None:
        raise RuntimeError(
            "with_poster needs a readable r_frame_rate to turn at_seconds into a frame index, "
            "and the source reports none."
        )
    num, den = rate
    frame = round(at_seconds * num / den)
    total = get_nb_frames(probe_json)
    if total is not None and total > 0:
        frame = min(frame, total - 1)
    derivative = Derivative(
        suffix=suffix,
        codec="mjpeg",
        pixel_format="yuvj420p",
        container="jpg",
        tap="source",
        # select first so the colour work runs on one frame. The scale converts
        # to BT.709 primaries at full range whatever the source matrix was;
        # zscale then applies the sRGB OETF to still-linear light.
        filters=(
            f"select=eq(n\\,{frame}),"
            "scale=out_color_matrix=bt709:out_range=full"
            ":flags=accurate_rnd+full_chroma_int+full_chroma_inp:sws_dither=none,"
            "zscale=t=iec61966-2-1:r=full"
        ),
        codec_params=("-q:v", str(quality)),
        # -update 1 is how image2 is told a bare filename means one image
        # rather than a malformed sequence pattern.
        output_flags=("-frames:v", "1", "-update", "1"),
    )
    return profile.replace(derivatives=(*profile.derivatives, derivative))
