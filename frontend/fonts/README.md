# Bundled GUI font

**Pretendard** — the primary UI sans, the same face the desktop trainer GUI
loads (`anima_lora/gui/theme.py::_load_bundled_fonts`). Bundling it rather than
riding on `system-ui` is what keeps the two tools looking like one product on
every OS, and it unifies the EN/KO look of paths and captions.

`PretendardVariable.subset.woff2` is the upstream **variable** font (axis `wght`
45–930) subset to Latin + Hangul + the punctuation and symbols the UI uses
(`… ⚠ ⚙ ✓ →`). Colour emoji like the picker's 📁 are not in Pretendard at all
and come from the system emoji family. One variable file at 1.7 MB beats three static weights at
2.0 MB *and* gives every weight in between, so 400/500/600/700 in `styles.css`
are all real instances, never synthesised.

`scripts/build_frontend.sh` inlines this file into
`anime_tools/gui/static/index.html` as a base64 `data:` URL — the built GUI is
one self-contained file with no font request at runtime, online or off. Han
ideographs (JA/ZH) and emoji are outside the subset and fall back to the system
families named in the `--font-ui` stack, exactly as the Qt GUI does.

To regenerate after an upstream bump:

```sh
curl -sLO https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/packages/pretendard/dist/web/variable/woff2/PretendardVariable.woff2
uvx --from 'fonttools[woff]' pyftsubset PretendardVariable.woff2 \
  --unicodes='U+0000-00FF,U+0131,U+0152-0153,U+2000-206F,U+2018-201D,U+2026,U+20AC,U+2122,U+2190-2193,U+2212,U+2713,U+26A0,U+2699,U+FEFF,U+FFFD,U+AC00-D7A3,U+1100-11FF,U+3130-318F' \
  --layout-features='' --flavor=woff2 --output-file=PretendardVariable.subset.woff2
```

## License

Pretendard is licensed under the **SIL Open Font License 1.1**.

- Upstream: https://github.com/orioncactus/pretendard
- Copyright © 2021 Kil Hyung-jin (길형진), with reserved font name "Pretendard".

The OFL permits bundling and redistribution with software (including as an
embedded `data:` URL). Keep this notice alongside the font file.
