# presentations/

Quarto + reveal.js decks. Render with:

```bash
quarto render aind-open-data-avi-inventory.qmd
quarto preview aind-open-data-avi-inventory.qmd   # live reload
```

## Encoding-settings comparison clips

`aind-open-data-avi-inventory.qmd` has a 3-up comparison grid showing the
chosen cascade setting (`slow / CRF 18`) bracketed by the next-step-up
(`veryslow / CRF 18`) and the next-step-down (`slow / CRF 20`). Expected
filenames in `clips/`:

- `cascade_veryslow_crf18.mp4`
- `cascade_slow_crf18.mp4` — chosen
- `cascade_slow_crf20.mp4`

All clips on the grid slide are paused, seeked to `t=0`, and replayed every
time the slide is opened (`Reveal.slidechanged` event in the qmd's
`include-after-body`). For sync to be tight:

- All clips should be the **same duration** and **same fps**.
- Encode `+faststart` (`-movflags +faststart`) so playback starts without
  waiting on metadata at the end.
- Browsers only honor `autoplay` on `muted` videos — keep `muted` set.

The clips shipped with this deck were cut from the cascade-benchmark stage-2
outputs at the fast-tongue-extension peak (testBottom, 8.4375-8.5625 s in the
15-second chunk, corresponding to ~28.95 s into the original raw AVI). 0.125 s
of source @ 522 fps is slowed 32× to 4 s @ 60 fps:

```bash
STAGE2=/mnt/Data/encodes/stage2
encode_clip() {
  local src=$1 dst=$2
  ffmpeg -y -hide_banner -v error \
    -ss 8.4375 -t 0.125 -i "$src" \
    -vf "setpts=32*PTS" -r 60 \
    -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
    -movflags +faststart -an \
    "$dst"
}
encode_clip "$STAGE2/testBottom_from_legacy_veryslow_crf18.mp4" clips/cascade_veryslow_crf18.mp4
encode_clip "$STAGE2/testBottom_from_legacy_slow_crf18.mp4"     clips/cascade_slow_crf18.mp4
encode_clip "$STAGE2/testBottom_from_legacy_slow_crf20.mp4"     clips/cascade_slow_crf20.mp4
```
