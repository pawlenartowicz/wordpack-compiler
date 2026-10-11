import io
import time

from wordpack_compiler.progress import Progress, format_count, format_size, format_time


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_off_terminal_update_prints_only_milestones():
    out = io.StringIO()
    p = Progress("label  ", out)
    p.update("quiet")
    assert out.getvalue() == ""
    p.update("10%", milestone=True)
    assert out.getvalue() == "label  10%\n"


def test_done_prints_one_line():
    out = io.StringIO()
    Progress("label  ", out).done("finished")
    assert out.getvalue() == "label  finished\n"


def test_stream_without_isatty_counts_as_not_a_terminal():
    class Bare:
        def __init__(self):
            self.text = ""

        def write(self, s):
            self.text += s

        def flush(self):
            pass

    out = Bare()
    p = Progress("x ", out)
    p.update("hidden")
    p.done("shown")
    assert out.text == "x shown\n"


def test_terminal_rewrites_in_place_and_pads_shorter_text():
    out = Tty()
    p = Progress("L ", out)
    p.update("a long line")
    time.sleep(0.15)
    p.update("short")
    assert out.getvalue() == "\rL a long line" + "\rL short" + " " * 6


def test_terminal_drops_update_inside_throttle_window():
    out = Tty()
    p = Progress("L ", out)
    p.update("first")
    p.update("second")
    assert out.getvalue() == "\rL first"


def test_terminal_done_overwrites_the_in_place_line():
    out = Tty()
    p = Progress("L ", out)
    p.update("a long line")
    p.done("ok")
    assert out.getvalue() == "\rL a long line" + "\rL ok" + " " * 9 + "\n"


def test_format_time():
    assert format_time(5) == "0:05"
    assert format_time(161) == "2:41"
    assert format_time(3723) == "1:02:03"


def test_format_size():
    assert format_size(67_000) == "67 KB"
    assert format_size(412_000_000) == "412 MB"
    assert format_size(4_120_000_000) == "4,120 MB"


def test_format_count():
    assert format_count(1291147) == "1,291,147"


def test_terminal_line_is_cut_to_one_less_than_the_width(monkeypatch):
    monkeypatch.setattr("shutil.get_terminal_size", lambda: __import__("os").terminal_size((10, 24)))
    out = Tty()
    p = Progress("L ", out)
    p.update("0123456789abc")
    time.sleep(0.15)
    p.update("xy")
    assert out.getvalue() == "\rL 0123456" + "\rL xy" + " " * 5
