"""CLI for batch transcoding videos with encoding profiles.

All settings can be stored in ``aind-transcode.toml`` in the working directory.
CLI arguments always take priority over the TOML file.

Example ``aind-transcode.toml``::

    input = ["side_camera_left.avi"]
    output_dir = "transcoded"
    profile = "offline-8bit"
    preset = "veryfast"
    overwrite = false
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from aind_video_utils.encoding import PROFILES, EncodingProfile
from aind_video_utils.probe import get_nb_frames, probe
from aind_video_utils.transcode import VIDEO_EXTENSIONS, transcode_video

ProfileName = Literal["offline-8bit", "offline-10bit", "online-8bit", "online-10bit"]


def _replace_codec_param(params: tuple[str, ...], flag: str, value: str) -> tuple[str, ...]:
    """Return *params* with the value following *flag* replaced."""
    lst = list(params)
    for i, item in enumerate(lst):
        if item == flag and i + 1 < len(lst):
            lst[i + 1] = value
            return tuple(lst)
    return params


class TranscodeSettings(BaseSettings):
    """Settings for ``aind-transcode``.  Can be stored in ``aind-transcode.toml``."""

    model_config = SettingsConfigDict(toml_file="aind-transcode.toml")

    input: CliPositionalArg[list[str]] = Field(
        description="Video files or directories to transcode.",
    )
    output_dir: str | None = Field(
        None,
        description="Output directory.  Defaults to same directory as input.",
    )
    profile: ProfileName = Field(
        "offline-8bit",
        description="Encoding profile: offline-8bit, offline-10bit, online-8bit, online-10bit.",
    )
    crf: int | None = Field(
        None,
        description="Override the profile's quality setting (-crf for offline, -cq for online).",
    )
    preset: str | None = Field(
        None,
        description="Override the profile's -preset value.",
    )
    no_auto_fix_colorspace: bool = Field(
        False,
        description="Skip automatic setparams probing for missing color metadata.",
    )
    overwrite: bool = Field(False, description="Re-encode even if output exists.")
    jobs: int = Field(
        default_factory=lambda: max(1, (os.cpu_count() or 1) // 2),
        description="Number of parallel transcodes.  Defaults to half the CPU count.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            CliSettingsSource(settings_cls, cli_parse_args=True),
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    def _resolve_profile(self) -> EncodingProfile:
        """Build the effective profile with any CLI overrides applied."""
        prof = PROFILES[self.profile]
        if self.preset is not None:
            prof = prof.replace(
                codec_params=_replace_codec_param(prof.codec_params, "-preset", self.preset),
            )
        if self.crf is not None:
            crf_str = str(self.crf)
            # Offline profiles use -crf, online profiles use -cq
            updated = _replace_codec_param(prof.codec_params, "-crf", crf_str)
            if updated is prof.codec_params:
                updated = _replace_codec_param(prof.codec_params, "-cq", crf_str)
            prof = prof.replace(codec_params=updated)
        return prof

    def cli_cmd(self) -> None:
        """Run the batch transcode with Rich progress bars."""
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskID,
            TextColumn,
            TimeElapsedColumn,
        )

        console = Console()

        # Check ffmpeg is available.
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise SystemExit("ffmpeg is not installed or not found in PATH.") from exc

        resolved = self._resolve_profile()

        # Collect input files.
        files: list[Path] = []
        for entry in self.input:
            p = Path(entry)
            if p.is_dir():
                for ext in VIDEO_EXTENSIONS:
                    files.extend(sorted(p.glob(f"*{ext}")))
            elif p.is_file():
                files.append(p)
            else:
                console.print(f"[yellow]Skipping (not found): {entry}[/yellow]")

        if not files:
            raise SystemExit("No video files found.")

        out_dir = Path(self.output_dir) if self.output_dir else None

        # Build work list, skipping files that already have output.
        work: list[tuple[Path, Path]] = []
        for path in files:
            dest_dir = out_dir or path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            output_path = dest_dir / f"{path.stem}.{resolved.container}"

            if output_path.exists() and not self.overwrite:
                console.print(f"  [dim]skip[/dim]  {path.name}  (output exists)")
            else:
                work.append((path, output_path))

        if not work:
            console.print("\nNothing to transcode.")
            return

        n_jobs = min(self.jobs, len(work))
        src_to_dst = dict(work)
        failed = 0

        # Probe all files for frame counts (fast, sequential).
        src_to_frames: dict[Path, int | None] = {}
        for src, _ in work:
            try:
                probe_json = probe(src)
                src_to_frames[src] = get_nb_frames(probe_json)
            except Exception:  # noqa: BLE001
                src_to_frames[src] = None

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="bold"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall = progress.add_task(
                f"Overall  (profile={self.profile}, jobs={n_jobs})",
                total=len(work),
            )

            # Per-file tasks with frame-level progress bars.
            file_tasks: dict[Path, TaskID] = {}
            for src, _ in work:
                total = src_to_frames[src]
                tid = progress.add_task(
                    f"  {src.name}",
                    total=total,
                    visible=False,
                )
                file_tasks[src] = tid

            def _run(src: Path, dst: Path) -> Path:
                tid = file_tasks[src]
                progress.update(tid, visible=True)

                def _on_frame(frame: int) -> None:
                    progress.update(tid, completed=frame)

                return transcode_video(
                    src,
                    dst,
                    profile=resolved,
                    auto_fix_colorspace=not self.no_auto_fix_colorspace,
                    on_progress=_on_frame,
                )

            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures = {
                    pool.submit(_run, src, dst): src for src, dst in work
                }

                for future in as_completed(futures):
                    src = futures[future]
                    dst = src_to_dst[src]
                    tid = file_tasks[src]
                    try:
                        future.result()
                        size_mb = dst.stat().st_size / (1024 * 1024)
                        progress.update(
                            tid,
                            description=f"  [green]ok[/green]  {src.name}  ({size_mb:.1f} MB)",
                            completed=progress.tasks[tid].total or 1,
                            total=progress.tasks[tid].total or 1,
                        )
                    except subprocess.CalledProcessError as exc:
                        failed += 1
                        stderr = exc.stderr
                        if isinstance(stderr, bytes):
                            stderr = stderr.decode(errors="replace")
                        progress.update(
                            tid,
                            description=f"  [red]FAIL[/red]  {src.name}",
                            completed=progress.tasks[tid].total or 1,
                            total=progress.tasks[tid].total or 1,
                        )
                        progress.console.print(f"    {(stderr or '')[:500]}")
                    progress.update(overall, advance=1)

        console.print(
            f"\n[green]Done.[/green]  {len(work) - failed}/{len(work)} succeeded"
        )


def main() -> None:
    """Entry point for aind-transcode CLI."""
    CliApp.run(TranscodeSettings)


if __name__ == "__main__":
    main()
