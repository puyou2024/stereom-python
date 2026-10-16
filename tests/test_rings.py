import networkx as nx
import numpy as np

from stereom.rings import analyze_rings, find_rings, ring_geometry


def _square_ring_graph(side: float = 100.0):
    G = nx.MultiGraph()
    positions = {
        0: (0, 0, 0),
        1: (side, 0, 0),
        2: (side, side, 0),
        3: (0, side, 0),
    }
    for n, p in positions.items():
        G.add_node(n, pos=p, voxel_degree=-1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, (a, b) in enumerate(edges):
        G.add_edge(a, b, key=i, id=i, path=np.zeros((0, 3), dtype=np.int64), node_a=a, node_b=b)
    return G


def test_find_rings_finds_single_square_once():
    G = _square_ring_graph()
    cycles = find_rings(G, min_size=4, max_size=7, seed_min_degree=2)
    keys = {tuple(sorted(c)) for c in cycles}
    assert keys == {(0, 1, 2, 3)}


def test_square_ring_area_and_diameter():
    side = 100.0
    G = _square_ring_graph(side=side)
    cycles = find_rings(G, min_size=4, max_size=7, seed_min_degree=2)
    ring = ring_geometry(G, cycles[0], voxel_size_um=1.0, flatness_r2_min=0.5)
    assert ring is not None
    assert np.isclose(ring.area_um2, side * side, rtol=1e-6)
    expected_d = 2.0 * np.sqrt((side * side) / np.pi)
    assert np.isclose(ring.d_ring_um, expected_d, rtol=1e-6)
    assert np.isclose(ring.flatness_r2, 1.0, atol=1e-9)


def test_nonplanar_ring_rejected_by_flatness_filter():
    # A ring whose 4 corners lie exactly in z=0 is a poor test of
    # non-planarity rejection: with only 4 points, a least-squares plane
    # has enough freedom to tilt and still fit a single outlier corner
    # reasonably well. A real ring's plane fit uses the dense branch-voxel
    # boundary (many points), which much more strongly resists a bad fit
    # when the boundary is genuinely warped -- reproduced here with
    # per-point z jitter along each edge.
    rng = np.random.default_rng(0)
    corners = {0: (0, 0, 0), 1: (100, 0, 0), 2: (100, 100, 0), 3: (0, 100, 0)}
    G = nx.MultiGraph()
    for n, p in corners.items():
        G.add_node(n, pos=p, voxel_degree=-1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    mag = 80.0
    for i, (a, b) in enumerate(edges):
        pa, pb = np.array(corners[a], dtype=float), np.array(corners[b], dtype=float)
        t = np.linspace(0, 1, 20, endpoint=False)[1:, None]
        seg = pa * (1 - t) + pb * t
        seg[:, 2] = rng.uniform(-mag, mag, size=seg.shape[0])
        G.add_edge(a, b, key=i, id=i, path=seg.astype(np.int64), node_a=a, node_b=b)

    cycles = find_rings(G, min_size=4, max_size=7, seed_min_degree=2)
    ring = ring_geometry(G, cycles[0], voxel_size_um=1.0, flatness_r2_min=0.5)
    assert ring is None


def test_cube_graph_finds_exactly_six_faces_no_duplicates():
    # 8 nodes of a cube, edges along the 12 cube edges -> exactly 6 square
    # faces (4-rings); diagonals are NOT edges, so no smaller/other rings
    # exist. This exercises dedup on a topology where a naive DFS
    # rediscovers each face many times from different seeds/directions.
    G = nx.MultiGraph()
    coords = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    for i, c in enumerate(coords):
        G.add_node(i, pos=tuple(50 * v for v in c), voxel_degree=-1)
    eid = 0
    for i, ci in enumerate(coords):
        for j, cj in enumerate(coords):
            if j <= i:
                continue
            diff = sum(abs(a - b) for a, b in zip(ci, cj))
            if diff == 1:  # differ in exactly one coordinate -> cube edge
                G.add_edge(i, j, key=eid, id=eid, path=np.zeros((0, 3), dtype=np.int64), node_a=i, node_b=j)
                eid += 1

    # Restricted to max_size=4 to isolate the 6 square faces from the
    # cube's other (non-planar) 6- and 8-node cycles.
    cycles = find_rings(G, min_size=4, max_size=4, seed_min_degree=3)
    keys = {tuple(sorted(c)) for c in cycles}
    assert len(keys) == 6
    for key in keys:
        assert len(key) == 4


def test_analyze_rings_returns_dataframe_with_expected_columns():
    G = _square_ring_graph()
    df = analyze_rings(G, voxel_size_um=1.0, min_size=4, max_size=7, seed_min_degree=2)
    assert len(df) == 1
    for col in ("size", "area_um2", "d_ring_um", "flatness_r2"):
        assert col in df.columns
