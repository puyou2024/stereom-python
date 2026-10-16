import numpy as np

from stereom.graph_build import build_graph, build_graph_from_node_mask, identify_nodes
from stereom.graph_clean import merge_clusters


def _double_y_skeleton(shape=(30, 30, 55)) -> np.ndarray:
    """Two 3-way junctions A=(10,10,10), B=(10,10,40) joined by a straight
    bridge along z, each junction also carrying two free (dangling) arms
    along +x and +y. Only the bridge has BOTH ends touching a registered
    node -- the four free arms must each be dropped (one end touches no
    node at all), exactly like a real whisker in the reference algorithm
    (see graph_build.py's module docstring).

    Because three orthogonal arms meet at a single voxel, several of the
    immediate neighbour voxels around each junction are themselves mutually
    26-adjacent and each independently satisfy k>=4 -- a real "fat
    junction cluster" (spec Sec 3.3/3.5: "a thick junction yields a
    cluster of qualifying voxels, not one node. Fixed later by merging --
    by design."). Tests below therefore check topology and approximate
    position/length, and use ``merge_clusters`` to consolidate each
    cluster to one node, exactly as the real pipeline always does
    immediately after ``build_graph``.
    """
    skel = np.zeros(shape, dtype=bool)
    skel[10, 10, 10] = True  # junction A
    skel[10, 10, 40] = True  # junction B
    for z in range(11, 40):
        skel[10, 10, z] = True  # bridge, interior voxels z=11..39 (29 voxels)
    for x in range(11, 26):
        skel[x, 10, 10] = True  # A's free +x arm
    for y in range(11, 26):
        skel[10, y, 10] = True  # A's free +y arm
    for x in range(11, 26):
        skel[x, 10, 40] = True  # B's free +x arm
    for y in range(11, 26):
        skel[10, y, 40] = True  # B's free +y arm
    return skel


def test_identify_nodes_finds_clusters_at_both_junctions_only():
    skel = _double_y_skeleton()
    node_mask, voxel_degree = identify_nodes(skel, degree_threshold=4)
    node_coords = np.argwhere(node_mask)
    # Every candidate node voxel must be within the immediate vicinity of
    # one of the two junctions (not scattered along the plain 1-voxel-wide
    # arms, which have k=3 everywhere and k=2 at their free tip).
    junctions = np.array([[10, 10, 10], [10, 10, 40]])
    dists = np.min(
        np.linalg.norm(node_coords[:, None, :].astype(float) - junctions[None, :, :], axis=-1),
        axis=1,
    )
    assert np.all(dists <= 2.0)
    assert node_coords.shape[0] >= 2  # at least the two true centres qualify
    assert voxel_degree[10, 10, 10] == 3
    assert voxel_degree[10, 10, 40] == 3


def test_build_graph_keeps_only_the_bridge_branch():
    skel = _double_y_skeleton()
    G, stats = build_graph(skel, degree_threshold=4)
    G = merge_clusters(G, d_m_voxels=2.0)

    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    (u, v, k, data), = G.edges(keys=True, data=True)
    positions = {G.nodes[u]["pos"], G.nodes[v]["pos"]}
    expected = {(10, 10, 10), (10, 10, 40)}
    # Registered node positions land on the specific blob voxel each
    # branch endpoint happened to touch, not necessarily the geometric
    # centre voxel -- within 2 voxels of it is the correct expectation.
    for p in positions:
        assert min(np.linalg.norm(np.array(p) - np.array(e)) for e in expected) <= 2.0
    assert 25 <= data["path"].shape[0] <= 29
    # 4 free arms dropped as dangling (one end never touches a node)
    assert stats.n_dropped_dangling == 4
    assert stats.n_branches_kept == 1


def test_isolated_straight_strut_produces_empty_graph():
    # No junction anywhere -> node_identification finds nothing -> no
    # branch can ever satisfy "both ends touch a registered node". This
    # matches the reference algorithm's actual behaviour (not a bug): a
    # free-floating strut with no junction is topologically inert to this
    # registration scheme in both the original and this port.
    shape = (10, 10, 60)
    skel = np.zeros(shape, dtype=bool)
    skel[5, 5, 5:55] = True
    G, stats = build_graph(skel, degree_threshold=4)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_single_voxel_branch_between_two_close_junctions():
    # Two junctions separated by exactly one non-node skeleton voxel --
    # exercises the "size==1 segment, must touch exactly 2 distinct node
    # voxels" special case in branch_sort2 / graph_build.build_graph.
    #
    # Node candidacy is specified directly (via build_graph_from_node_mask)
    # rather than derived from a rasterized multi-arm junction: any real
    # >=3-arm junction's immediate neighbours are mutually 26-adjacent and
    # tend to pull in neighbouring voxels (including, for closely-spaced
    # junctions, the intended single bridging voxel itself) as additional
    # node candidates -- a real emergent property of the algorithm (see
    # test_identify_nodes_finds_clusters_at_both_junctions_only), just not
    # one that leaves room to hand-place an isolated single-voxel bridge
    # between two clean single-voxel nodes.
    # node_mask is specified directly, so skel need only contain the
    # bridging voxel itself (plus the two node voxels, though those are
    # subtracted out when forming segments regardless) -- no "arm" voxels
    # are needed here, and including any would only risk them 26-adjacently
    # merging into the same segment as the bridge (as an earlier version
    # of this test discovered the hard way).
    shape = (20, 20, 20)
    skel = np.zeros(shape, dtype=bool)
    skel[10, 10, 10] = True  # A
    skel[10, 10, 12] = True  # B
    skel[10, 10, 11] = True  # the single bridging voxel

    node_mask = np.zeros(shape, dtype=bool)
    node_mask[10, 10, 10] = True
    node_mask[10, 10, 12] = True

    G, stats = build_graph_from_node_mask(skel, node_mask)
    assert G.number_of_edges() == 1
    (u, v, k, data), = G.edges(keys=True, data=True)
    assert data["path"].shape[0] == 1
    assert tuple(data["path"][0].tolist()) == (10, 10, 11)
