"""Point-cloud DASH manifest parsing, size helpers, and the origin Server.

Moved out of the legacy network_model.py. Logic for parsing, size conversion, and
serving point clouds is unchanged; `PointCloudServer` is renamed to `Server`
(with a back-compat alias) and gains a `backhaul_links` registry so it can be a
first-class node in the topology (reachable only via a BackhaulLink).
"""

import math
import xml.etree.ElementTree as ET

# Default quality-utility endpoints (used only when a manifest carries no
# densities). Derive real endpoints per manifest via manifest_quality_endpoints.
DEFAULT_Q_LOW_DENSITY = 30000.0
DEFAULT_Q_HIGH_DENSITY = 1060000.0


def density_quality(density, low_density, high_density):
    """Normalized [0,1] quality utility from a representation's point density.

    Log-density scaling so a large density ratio doesn't dwarf stall/switch
    penalties. This is THE quality signal shared by the RL reward
    (src/rl/features.quality delegates here) and the quality-aware QoE.
    """
    lo = math.log10(low_density if low_density else DEFAULT_Q_LOW_DENSITY)
    hi = math.log10(high_density if high_density else DEFAULT_Q_HIGH_DENSITY)
    if hi <= lo:
        return 0.0
    d = max(1.0, float(density or 1))
    return max(0.0, min(1.0, (math.log10(d) - lo) / (hi - lo)))


def manifest_quality_endpoints(frames):
    """(min_density, max_density) across all reps of all frames of a manifest,
    so the ladder's lowest rep maps to ~0.0 and the highest to ~1.0."""
    densities = [r['density'] for fr in frames for r in fr['representations']
                 if r.get('density')]
    if not densities:
        return DEFAULT_Q_LOW_DENSITY, DEFAULT_Q_HIGH_DENSITY
    lo, hi = min(densities), max(densities)
    return (lo, hi) if hi > lo else (DEFAULT_Q_LOW_DENSITY, DEFAULT_Q_HIGH_DENSITY)


def parse_mpd_xml(xml_path):
    """Parse MPD XML file (supports both old format and new G-PCC format)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    frames = []
    for frame_elem in root.findall('Frame'):
        frame_id = int(frame_elem.attrib['id'])
        reps = []
        for rep_elem in frame_elem.find('AdaptationSet').findall('Representation'):
            rep = {
                'id': int(rep_elem.attrib['id']),
                'density': int(rep_elem.attrib.get('density', 100)),
                'size': rep_elem.attrib['size'],
                'base_url': rep_elem.find('BaseURL').text,
                'bandwidth': int(rep_elem.attrib.get('bandwidth', 0)),
                'quality': rep_elem.attrib.get('quality', 'unknown'),
                'geom_qp': int(rep_elem.attrib.get('geomQP', 0)),
                'attr_qp': int(rep_elem.attrib.get('attrQP', 0)),
                # Real G-PCC coded bitstream size (bytes) when produced by
                # gpcc/encode_frames.py; None => fall back to the analytical model.
                'coded_bytes': (float(rep_elem.attrib['codedBytes'])
                                if 'codedBytes' in rep_elem.attrib else None),
            }
            reps.append(rep)
        frames.append({'id': frame_id, 'representations': reps})
    return frames


def size_to_bytes(size_str):
    """Convert size string (e.g., '1.05M', '419.3K') to bytes."""
    if size_str.endswith('M'):
        return float(size_str[:-1]) * 1024 * 1024
    if size_str.endswith('K'):
        return float(size_str[:-1]) * 1024
    return float(size_str)


def size_to_bits(size_str):
    """Convert size string to bits."""
    return size_to_bytes(size_str) * 8


# --- Coded (compressed) bitrate model ---
# Raw .ply sizes are NOT streaming bitrates. In real DASH-PC each frame is
# pre-encoded with a point-cloud codec (MPEG G-PCC / V-PCC) and the compressed
# bitstream is streamed. We model the coded size analytically as
# density * bits_per_point (a standard G-PCC rate parameterization; bpp ~ codec
# quality / QP). At ~3 bpp the three density tiers form a clean bitrate ladder
# over the 4G traces at the true 30 fps (low~2.7 / med~14 / high~95 Mbps).
# To use REAL coded sizes instead, encode the .ply files with TMC13/TMC2 and put
# the bitstream sizes (or a per-rep bandwidth) in the manifest, then read them here.
DEFAULT_BITS_PER_POINT = 3.0


def coded_size_bytes(rep, bits_per_point=None):
    """Compressed (streamed) size of one representation's frame, in bytes.

    Prefers the REAL G-PCC coded size from the manifest (`coded_bytes`); falls
    back to the analytical density * bits_per_point model when unavailable.
    """
    real = rep.get('coded_bytes')
    if real:
        return float(real)
    bpp = DEFAULT_BITS_PER_POINT if bits_per_point is None else bits_per_point
    density = max(1, int(rep.get('density', 1)))
    return density * bpp / 8.0


def coded_bitrate_bps(rep, bits_per_point=None, fps=30.0):
    """Streamed bitrate (bps) of a representation at the given fps.

    Prefers the manifest's real `bandwidth`; else derives it from the coded size.
    """
    bw = rep.get('bandwidth')
    if bw:
        return float(bw)
    return coded_size_bytes(rep, bits_per_point) * 8.0 * fps


class PointCloud:
    def __init__(self, points, attributes=None, meta=None):
        self.points = points
        self.attributes = attributes or {}
        self.meta = meta or {}


class DASHPCManifest:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_representations(self, frame_id):
        for frame in self.frames:
            if frame['id'] == frame_id:
                return frame['representations']
        return []


class Server:
    """First-class origin node: holds the manifest + point clouds and serves a
    representation when an edge asks. It never references users or edges; it is
    reachable only through BackhaulLink objects registered on it.
    """

    def __init__(self, base_url):
        self.base_url = base_url
        self.pointclouds = {}
        self.manifest = DASHPCManifest()
        self.backhaul_links = []

    def add_pointcloud(self, frame_id, representation_id, pointcloud):
        if frame_id not in self.pointclouds:
            self.pointclouds[frame_id] = {}
        self.pointclouds[frame_id][representation_id] = pointcloud

    def get_manifest(self):
        return self.manifest

    def serve_pointcloud(self, frame_id, representation_id):
        return self.pointclouds[frame_id][representation_id]

    def register_backhaul(self, link):
        """Register a BackhaulLink that connects an edge to this server."""
        self.backhaul_links.append(link)


# Back-compat alias: existing code referred to the origin as PointCloudServer.
PointCloudServer = Server
