"""
Unit tests for the centralized configuration loader.
"""

from src.config import _deep_merge, _dict_to_namespace, _env_overrides, load_config


class TestDeepMerge:
    def test_nested_dicts_merged_recursively(self):
        base = {"model": {"elo_k": 20.0, "elo_draw_width": 0.44}}
        override = {"model": {"elo_k": 25.0}}
        merged = _deep_merge(base, override)
        assert merged["model"]["elo_k"] == 25.0
        assert merged["model"]["elo_draw_width"] == 0.44

    def test_new_keys_added(self):
        base = {"model": {"elo_k": 20.0}}
        override = {"new_section": {"foo": 1}}
        merged = _deep_merge(base, override)
        assert merged["new_section"]["foo"] == 1

    def test_scalar_override(self):
        merged = _deep_merge(
            {"data": {"data_dir": "data/raw"}}, {"data": {"data_dir": "custom"}}
        )
        assert merged["data"]["data_dir"] == "custom"


class TestEnvOverrides:
    def test_env_var_parsed(self, monkeypatch):
        monkeypatch.setenv("FOOTBALL_PREDICTOR__model__elo_k", "30.0")
        overrides = _env_overrides()
        assert overrides["model"]["elo_k"] == 30.0

    def test_env_var_ignores_unrelated(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        assert _env_overrides() == {}

    def test_env_var_string_stays_string(self, monkeypatch):
        monkeypatch.setenv("FOOTBALL_PREDICTOR__model__name", "my-model")
        overrides = _env_overrides()
        assert overrides["model"]["name"] == "my-model"


class TestDictToNamespace:
    def test_nested_dict_becomes_nested_namespace(self):
        ns = _dict_to_namespace({"model": {"elo_k": 20.0}, "list": [1, 2]})
        assert ns.model.elo_k == 20.0
        assert ns.list == [1, 2]


class TestLoadConfig:
    def test_loads_project_config(self):
        cfg = load_config()
        assert hasattr(cfg, "model")
        assert hasattr(cfg, "data")
        # config.yaml should contain the expected model settings
        assert hasattr(cfg.model, "elo_k")
        assert hasattr(cfg.model, "dc_xi")

    def test_loads_from_custom_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  elo_k: 99.0\n")
        cfg = load_config(config_file)
        assert cfg.model.elo_k == 99.0

    def test_missing_file_returns_defaults(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_config(missing)
        assert cfg.model.elo_k == 20.0
        assert cfg.model.dc_xi == 0.0018

    def test_env_override_wins_over_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model:\n  elo_k: 10.0\n")
        monkeypatch.setenv("FOOTBALL_PREDICTOR__model__elo_k", "45.0")
        cfg = load_config(config_file)
        assert cfg.model.elo_k == 45.0
