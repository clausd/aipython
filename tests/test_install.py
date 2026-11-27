# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import sys
from types import SimpleNamespace

from aipython import install


def test_package_names():
    assert install.package_for("sklearn.linear_model") == "scikit-learn"
    assert install.package_for("polars") == "polars"


def test_hint_for_missing_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    error = ModuleNotFoundError("No module named 'sklearn'", name="sklearn")
    assert "%ai install scikit-learn" in install.hint(error)
    assert "uv tool" not in install.hint(error)

    (tmp_path / "uv-receipt.toml").write_text("")
    assert "uv tool install aipython-shell --with scikit-learn" in install.hint(error)


def test_no_hint_for_other_errors():
    assert install.hint(ValueError("x")) is None
    assert install.hint(None) is None
    # the package exists, only a submodule is missing
    assert install.hint(ModuleNotFoundError("x", name="pandas.nope")) is None


def test_install_prefers_uv(monkeypatch):
    monkeypatch.setattr(install.shutil, "which", lambda name: "/bin/uv")
    assert install.install_command(["polars"]) == [
        "/bin/uv", "pip", "install", "--python", sys.executable, "polars"]
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert install.install_command(["polars"]) == [sys.executable, "-m", "pip", "install", "polars"]


def test_install_reports_failure():
    assert install.install(["x"], run=lambda cmd: SimpleNamespace(returncode=0))
    assert not install.install(["x"], run=lambda cmd: SimpleNamespace(returncode=1))
