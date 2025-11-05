from nerd.utils.config import load_config
from nerd.utils.logging import get_logger, setup_logger


def test_load_config_reads_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "alpha: 1\n"
        "beta:\n"
        "  - name: item\n"
        "    value: true\n"
    )

    data = load_config(cfg_path)
    assert data["alpha"] == 1
    assert isinstance(data["beta"], list)
    assert data["beta"][0]["value"] is True


def test_setup_logger_creates_file(tmp_path):
    log_path = tmp_path / "nerd.log"
    logger = setup_logger(logfile=log_path, verbose=True)
    child = get_logger("nerd.tests")

    child.debug("debug message")
    child.info("info message")

    for handler in logger.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()

    assert log_path.is_file()
    contents = log_path.read_text()
    assert "info message" in contents
