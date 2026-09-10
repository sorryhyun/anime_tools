"""The GUI launcher: what it points at, and what each platform's file says.

The write itself only runs on this platform — a ``.lnk`` needs Windows' shell
COM object, so the two POSIX writers are tested directly instead.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from anime_tools import shortcut as S


def test_gui_command_prefers_the_console_script_beside_the_interpreter(
    tmp_path, monkeypatch
):
    """``anime-tools-gui`` lives in the tool environment this runs out of; PATH
    is only the fallback for a venv whose scripts sit elsewhere."""
    suffix = ".exe" if sys.platform == "win32" else ""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / f"{S.GUI_SCRIPT}{suffix}"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(S.sys, "executable", str(bin_dir / "python"))
    assert S.gui_command() == [str(script), "--open"]


def test_gui_command_falls_back_to_the_module(tmp_path, monkeypatch):
    monkeypatch.setattr(S.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(S.shutil, "which", lambda _name: None)
    assert S.gui_command() == [
        str(tmp_path / "python"),
        "-m",
        "anime_tools.gui",
        "--open",
    ]


def test_command_file_cds_to_the_folder_it_pins(tmp_path):
    """macOS: Finder runs a ``.command`` from the user's home, so the ``cd`` is
    what makes the launcher open *this* dataset."""
    path = tmp_path / "launch.command"
    S._write_command(path, tmp_path, ["/opt/bin/anime-tools-gui", "--open"])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/bin/sh"
    assert f"cd {shlex.quote(str(tmp_path))}" in lines[2]
    assert lines[3] == "exec /opt/bin/anime-tools-gui --open"
    assert path.stat().st_mode & 0o111  # double-clickable means executable


def test_desktop_entry_pins_path_and_keeps_a_terminal(tmp_path):
    S._write_desktop(tmp_path / "l.desktop", tmp_path, ["/opt/bin/gui", "--open"])
    body = (tmp_path / "l.desktop").read_text(encoding="utf-8")
    assert body.startswith("[Desktop Entry]")
    assert f"Path={tmp_path}" in body
    assert 'Exec="/opt/bin/gui" "--open"' in body
    assert "Terminal=true" in body  # the server log is the window


def test_desktop_quoting_is_the_spec_not_the_shell():
    """A Desktop Entry escapes with backslashes inside double quotes; POSIX
    single-quoting (what shlex would emit) is not valid there."""
    assert S._desktop_quote(["/a b/gui", "$HOME"]) == '"/a b/gui" "\\$HOME"'


def test_create_shortcut_writes_one_and_rewrites_it(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gui_command", lambda: ["/opt/bin/gui", "--open"])
    if sys.platform == "win32" and not S.shutil.which("powershell"):
        pytest.skip("no powershell to author a .lnk")
    path = S.create_shortcut(tmp_path)
    assert path == S.launcher_path(tmp_path.resolve())
    assert path.is_file()
    assert S.create_shortcut(tmp_path) == path  # regenerating is not an error


def test_create_shortcut_refuses_a_non_directory(tmp_path):
    with pytest.raises(S.ShortcutError):
        S.create_shortcut(tmp_path / "nope")


def test_main_prints_the_path_the_installers_parse(tmp_path, capsys, monkeypatch):
    """Both installers read ``launcher: <path>`` off stdout to name it in the
    closing message."""
    monkeypatch.setattr(S, "create_shortcut", lambda d: Path(d or ".") / "x.command")
    assert S.main([str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == f"launcher: {tmp_path / 'x.command'}"


def test_main_reports_a_failure_without_a_traceback(capsys):
    assert S.main(["/definitely/not/here"]) == 1
    assert "could not create the launcher" in capsys.readouterr().err
