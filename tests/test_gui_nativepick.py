"""The two gestures that reach the host's desktop: the `…` on a path field
(``POST /api/pick``) and the ↗ beside a name (``POST /api/reveal``). Which
command :mod:`anime_tools.gui.nativepick` builds for a desktop, and what the two
routes do with it. Nothing is opened: ``_argv`` / ``_reveal_argv`` are inspected
rather than run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_tools.gui import nativepick as NP

# ---- which chooser this desktop has ---------------------------------------


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(NP.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)


def _which(monkeypatch, found: dict[str, str]):
    monkeypatch.setattr(NP.shutil, "which", lambda name: found.get(name))


def test_zenity_opens_in_the_directory_it_was_given(linux, monkeypatch):
    _which(monkeypatch, {"zenity": "/usr/bin/zenity"})
    argv = NP._argv("dir", Path("/data/set"), "Choose a path")
    assert argv[:2] == ["/usr/bin/zenity", "--file-selection"]
    assert "--directory" in argv
    # The trailing separator is what opens *in* the directory.
    assert f"--filename=/data/set{NP.os.sep}" in argv
    assert "--title=Choose a path" in argv


def test_a_file_field_asks_for_a_file(linux, monkeypatch):
    _which(monkeypatch, {"zenity": "/usr/bin/zenity"})
    assert "--directory" not in NP._argv("file", None, "t")


def test_kdialog_is_the_second_choice(linux, monkeypatch):
    _which(monkeypatch, {"kdialog": "/usr/bin/kdialog"})
    argv = NP._argv("dir", Path("/data"), "t")
    assert argv[:3] == ["/usr/bin/kdialog", "--getexistingdirectory", "/data"]


def test_no_chooser_and_no_display_are_both_nothing(linux, monkeypatch):
    _which(monkeypatch, {})
    assert NP._argv("dir", None, "t") is None and not NP.available()
    _which(monkeypatch, {"zenity": "/usr/bin/zenity"})
    monkeypatch.delenv("DISPLAY", raising=False)
    assert NP._argv("dir", None, "t") is None


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(NP.sys, "platform", "win32")
    monkeypatch.setattr(NP.shutil, "which", lambda n: rf"C:\ps\{n}.exe")


def test_the_windows_dialog_is_owned_so_it_lands_on_top(windows):
    """An unowned dialog opens behind the browser; the topmost owner puts it in
    front, owns both kinds, and is closed on the way out of either."""
    for kind, dialog in (("dir", "FolderBrowserDialog"), ("file", "OpenFileDialog")):
        script = NP._argv(kind, None, "t")[-1]
        assert dialog in script
        assert f"{NP.OWNER_VAR}.TopMost = $true" in script
        assert f"ShowDialog({NP.OWNER_VAR})" in script
        # Raised before the dialog and disposed of after, so a cancel leaks no
        # transparent topmost window.
        assert script.index(f"{NP.OWNER_VAR}.Show()") < script.index("ShowDialog(")
        assert script.endswith(f"{NP.OWNER_VAR}.Close()")


def test_windows_needs_a_powershell_to_run_the_dialog(monkeypatch):
    monkeypatch.setattr(NP.sys, "platform", "win32")
    monkeypatch.setattr(NP.shutil, "which", lambda n: None)
    assert NP._argv("dir", None, "t") is None


def test_scripted_backends_quote_their_arguments():
    # A path is data, and both of these reach a script rather than an argv.
    assert NP._quote_as("applescript", 'a"b\\c') == '"a\\"b\\\\c"'
    assert NP._quote_as("powershell", "a'b") == "'a''b'"


def test_a_title_is_trimmed_to_something_a_title_bar_holds():
    assert NP._clean_title("") == "anime_tools"
    assert len(NP._clean_title("x" * 500)) == NP.MAX_TITLE
    assert NP._clean_title("a\x00b\nc") == "abc"


def test_pick_reports_a_host_with_no_chooser(monkeypatch):
    monkeypatch.setattr(NP, "_argv", lambda *a: None)
    assert NP.pick("dir") == NP.Pick(None, available=False)


def test_a_cancel_is_not_an_unavailable_host(monkeypatch):
    monkeypatch.setattr(NP, "_argv", lambda *a: ["irrelevant"])

    class _Proc:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(NP.subprocess, "run", lambda *a, **k: _Proc())
    assert NP.pick("dir") == NP.Pick(None, available=True)


# ---- the route ------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    (tmp_path / "image_dataset").mkdir()
    return tmp_path


def _client(home, *, host: str = "127.0.0.1"):
    from fastapi.testclient import TestClient

    from anime_tools.gui.jobs import JobManager
    from anime_tools.gui.server import create_app

    app = create_app(jobs=JobManager(log_dir=home / "logs"), schemas={})
    return TestClient(app, client=(host, 4242))


@pytest.fixture
def calls(monkeypatch):
    """Stand in for the dialog: record the ask, answer with what was staged."""
    seen: list[tuple] = []
    answer: list[NP.Pick] = [NP.Pick(None, available=True)]

    def fake(kind, start=None, *, title=""):
        seen.append((kind, start, title))
        return answer[0]

    from anime_tools.gui import server as SV

    monkeypatch.setattr(SV.NP, "pick", fake)
    return seen, answer


def test_a_pick_comes_back_relative_to_the_home(home, calls):
    seen, answer = calls
    answer[0] = NP.Pick(str(home / "image_dataset" / "sub"), available=True)
    with _client(home) as c:
        body = c.post("/api/pick", json={"kind": "dir", "path": "image_dataset"}).json()
    assert body == {"available": True, "path": "image_dataset/sub"}
    kind, start, _ = seen[0]
    assert kind == "dir" and start == home / "image_dataset"


def test_a_pick_outside_the_home_stays_absolute(home, calls):
    _, answer = calls
    answer[0] = NP.Pick("/elsewhere/set", available=True)
    with _client(home) as c:
        body = c.post("/api/pick", json={"kind": "dir", "path": ""}).json()
    assert body["path"] == "/elsewhere/set"


def test_the_chooser_opens_at_the_first_directory_that_exists(home, calls):
    """A root the user is about to create is the ordinary case."""
    seen, _ = calls
    with _client(home) as c:
        c.post("/api/pick", json={"kind": "file", "path": "image_dataset/no/such"})
    kind, start, _ = seen[0]
    assert kind == "file" and start == home / "image_dataset"


def test_a_stranger_browses_only_the_dataset(home):
    """The fallback browser walks the machine only for a local client; a remote
    one sees the dataset trees and no more."""
    with _client(home, host="10.0.0.7") as c:
        assert c.get("/api/ls").json()["parent"] is None  # no way up
        assert c.get("/api/ls", params={"path": str(home.parent)}).status_code == 404
    with _client(home) as c:
        assert c.get("/api/ls").json()["parent"] == home.parent.as_posix()


def test_a_browser_on_another_machine_gets_the_fallback(home, calls):
    """A remote browser gets the fallback: the dialog would open server-side."""
    seen, _ = calls
    with _client(home, host="10.0.0.7") as c:
        body = c.post("/api/pick", json={"kind": "dir", "path": ""}).json()
    assert body == {"available": False, "path": None} and not seen


# ---- reveal: the ↗ beside a name ------------------------------------------


def test_a_folder_is_opened_and_a_file_is_revealed(monkeypatch, tmp_path):
    """The two are different commands: `open` shows the folder, `open -R` shows
    the file *inside* its folder rather than handing it to Preview."""
    monkeypatch.setattr(NP.sys, "platform", "darwin")
    f = tmp_path / "a.png"
    f.write_bytes(b"")
    assert NP._reveal_argv(tmp_path) == ["open", str(tmp_path)]
    assert NP._reveal_argv(f) == ["open", "-R", str(f)]


def test_explorer_takes_select_as_one_token(monkeypatch, tmp_path):
    monkeypatch.setattr(NP.sys, "platform", "win32")
    monkeypatch.setattr(NP.shutil, "which", lambda n: rf"C:\w\{n}.exe")
    f = tmp_path / "a.png"
    f.write_bytes(b"")
    argv = NP._reveal_argv(f)
    assert argv[1] == f"/select,{f}" and " " not in argv[1][:8]
    assert NP._reveal_argv(tmp_path)[1] == str(tmp_path)


def test_a_file_goes_to_the_file_manager_not_to_an_image_viewer(
    linux, monkeypatch, tmp_path
):
    """``xdg-open`` on a PNG opens the PNG; only ShowItems selects it in a
    folder, so the bus call wins for a file and xdg-open gets the parent."""
    f = tmp_path / "a.png"
    f.write_bytes(b"")
    _which(
        monkeypatch,
        {"dbus-send": "/usr/bin/dbus-send", "xdg-open": "/usr/bin/xdg-open"},
    )
    argv = NP._reveal_argv(f)
    assert argv[0] == "/usr/bin/dbus-send" and f"array:string:{f.as_uri()}" in argv
    _which(monkeypatch, {"xdg-open": "/usr/bin/xdg-open"})
    assert NP._reveal_argv(f) == ["/usr/bin/xdg-open", str(tmp_path)]
    assert NP._reveal_argv(tmp_path) == ["/usr/bin/xdg-open", str(tmp_path)]


def test_a_headless_host_reveals_nothing(linux, monkeypatch):
    _which(monkeypatch, {})
    assert not NP.can_reveal()
    monkeypatch.delenv("DISPLAY", raising=False)
    _which(monkeypatch, {"xdg-open": "/usr/bin/xdg-open"})
    assert not NP.can_reveal()


def test_explorer_exiting_nonzero_is_still_a_reveal(monkeypatch, tmp_path):
    """`explorer` returns 1 on success, so the return code is never read."""
    monkeypatch.setattr(NP.sys, "platform", "darwin")

    class _Proc:
        returncode = 1

    monkeypatch.setattr(NP.subprocess, "run", lambda *a, **k: _Proc())
    assert NP.reveal(tmp_path) is True


def test_a_desktop_that_never_answers_is_a_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(NP.sys, "platform", "darwin")

    def _boom(*a, **k):
        raise NP.subprocess.TimeoutExpired("open", NP.REVEAL_TIMEOUT_S)

    monkeypatch.setattr(NP.subprocess, "run", _boom)
    assert NP.reveal(tmp_path) is False


@pytest.fixture
def revealed(monkeypatch):
    """Record what the route handed the desktop, without opening anything."""
    seen: list[Path] = []
    from anime_tools.gui import server as SV

    monkeypatch.setattr(SV.NP, "reveal", lambda p: (seen.append(p), True)[1])
    monkeypatch.setattr(SV.NP, "can_reveal", lambda: True)
    return seen


def test_reveal_resolves_against_the_home(home, revealed):
    with _client(home) as c:
        assert c.post("/api/reveal", json={"path": "image_dataset"}).json() == {
            "revealed": True
        }
    assert revealed == [home / "image_dataset"]


def test_reveal_stays_inside_the_dataset(home, revealed):
    """The same containment rule as every other read: `..` is collapsed first,
    so a path that climbs out is a 404 and never reaches the desktop."""
    with _client(home) as c:
        assert (
            c.post("/api/reveal", json={"path": "image_dataset/../.."}).status_code
            == 404
        )
        # Inside, but not there: nothing to show.
        assert c.post("/api/reveal", json={"path": "nope"}).status_code == 404
    assert not revealed


def test_only_this_machine_may_open_a_window_on_it(home, revealed):
    """The window opens where the server is, so a remote browser is refused --
    and told so before it draws a button, by ``/api/info``."""
    with _client(home, host="10.0.0.7") as c:
        assert c.post("/api/reveal", json={"path": "image_dataset"}).status_code == 403
        assert c.get("/api/info").json()["can_reveal"] is False
    assert not revealed
    with _client(home) as c:
        assert c.get("/api/info").json()["can_reveal"] is True


def test_a_headless_host_says_so_to_a_local_browser(home, monkeypatch):
    from anime_tools.gui import server as SV

    monkeypatch.setattr(SV.NP, "can_reveal", lambda: False)
    with _client(home) as c:
        assert c.get("/api/info").json()["can_reveal"] is False
