import json

from stereom.config import StereomConfig


def test_config_roundtrip_json(tmp_path):
    config = StereomConfig(voxel_size_um=0.42, ring_max_size=6)
    path = tmp_path / "config.json"
    config.save(path)
    loaded = StereomConfig.load(path)
    assert loaded.voxel_size_um == 0.42
    assert loaded.ring_max_size == 6
    assert loaded.axes == ("L", "C", "R")


def test_config_defaults_match_spec_registry():
    config = StereomConfig()
    assert config.voxel_size_um == 0.65
    assert config.node_degree_threshold == 4
    assert config.degree2_guard_voxels == 5.0
    assert config.thickness_min_path_voxels == 10
    assert config.ring_max_size == 7
    assert config.fit_len_min_um == 2.0
    assert config.fit_len_max_um == 35.0
    assert config.fit_max_abs_z_um == 25.0
