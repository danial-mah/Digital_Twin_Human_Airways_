"""
pages/10_3D_VR_Viewer.py
========================
WebXR-ready interactive 3D viewer for the airway digital twin.

Renders geometry.stl as a pressure-coloured solid mesh.  Optionally overlays
the pressure/deformation field as a coloured point cloud.

Desktop  : orbit/zoom/pan with mouse.
VR headset: click the 'Enter VR' button (Meta Quest browser, Chrome+OpenXR, etc.)
           The page must be served over HTTPS or localhost for WebXR to activate.

A standalone HTML download is provided so the file can be opened directly
in a VR headset browser without needing Streamlit.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_io import (
    STRIDE,
    load_pressure_snapshot,
    load_ref_coords,
    load_snapshot_coords,
)
from core.i18n import t, lang_selector

st.set_page_config(page_title="3D / VR Viewer", page_icon="🥽", layout="wide")
lang_selector()
st.title(t("vr_title"))
st.caption(t("vr_caption"))

# ── STL loader ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_stl_verts(stl_path: str) -> np.ndarray:
    """Binary STL → vertex positions (n_tri*3, 3) float32, one row per vertex."""
    with open(stl_path, "rb") as f:
        f.read(80)
        n_tri = struct.unpack("<I", f.read(4))[0]
        dtype = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0",     "<f4", (3,)),
            ("v1",     "<f4", (3,)),
            ("v2",     "<f4", (3,)),
            ("attr",   "<u2"),
        ])
        tris = np.frombuffer(f.read(n_tri * 50), dtype=dtype)
    verts = np.empty((n_tri * 3, 3), dtype=np.float32)
    verts[0::3] = tris["v0"]
    verts[1::3] = tris["v1"]
    verts[2::3] = tris["v2"]
    return verts


@st.cache_data(show_spinner=False)
def get_mesh_pressure_norm(snap: int, stride: int, stl_path_str: str) -> np.ndarray:
    """Normalized pressure [0,1] for each STL vertex via nearest-neighbour lookup."""
    from scipy.spatial import KDTree
    verts    = load_stl_verts(stl_path_str)
    coords   = load_snapshot_coords(snap, stride)
    pressure = load_pressure_snapshot(snap, stride)
    _, idx   = KDTree(coords).query(verts, k=1)
    p_mesh   = pressure[idx]
    p_min, p_max = float(p_mesh.min()), float(p_mesh.max())
    return ((p_mesh - p_min) / max(p_max - p_min, 1e-9)).astype(np.float32)


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header(t("sidebar_settings"))

    snap       = st.slider(t("lbl_snapshot"), 1, 100, 1)
    colorscale = st.selectbox(t("lbl_colour_map"), ["Jet", "Viridis", "RdBu_r", "Plasma"], index=0)
    opacity    = st.slider(t("lbl_opacity"), 0.2, 1.0, 0.85, step=0.05)
    stride_ui  = st.selectbox(t("lbl_resolution"), [50, 20, 10], index=0,
                               help="Lower stride = more nodes = slower render")

    st.subheader(t("vr_layers"))
    show_mesh  = st.checkbox(t("vr_show_mesh"),  value=True)
    show_cloud = st.checkbox(t("vr_show_cloud"), value=False)

    if show_cloud:
        color_mode = st.radio(
            t("lbl_colour_by"),
            [t("vr_colour_pres"), t("vr_colour_z"), t("vr_colour_idx")],
            index=0,
        )
        point_size = st.slider(t("lbl_point_size"), 1, 8, 3)
    else:
        color_mode = t("vr_colour_pres")
        point_size = 3

    bg_dark = st.checkbox(t("vr_dark_bg"), value=True)

    st.divider()
    st.markdown(
        "**Orbit mode**\n"
        "- Left-drag: orbit\n"
        "- Scroll: zoom\n"
        "- Right-drag: pan\n\n"
        "**Fly mode** (press `F` or click button)\n"
        "- Left-drag: look around\n"
        "- WASD: fly forward/back/strafe\n"
        "- Q / E: rise / descend\n"
        "- Scroll: adjust speed"
    )
    st.caption(
        "VR: open on Meta Quest browser or connect a PC headset "
        "via SteamVR/OpenXR, then click **Enter VR**."
    )

# ── Load data ─────────────────────────────────────────────────────────────────

stl_path  = Path(__file__).parent.parent / "geometry.stl"
stl_verts = load_stl_verts(str(stl_path)) if stl_path.exists() else None

with st.spinner(f"Loading snapshot {snap}…"):
    try:
        coords   = load_snapshot_coords(snap, stride_ui)
        pressure = load_pressure_snapshot(snap, stride_ui)
    except FileNotFoundError:
        coords   = load_ref_coords(stride_ui)
        pressure = np.zeros(len(coords))
        st.warning(t("vr_warn_missing"))

n_pts = len(coords)

# ── Normalise to [-1, 1] with shared scale so mesh + cloud align ─────────────

all_vals = np.concatenate([coords.ravel(), stl_verts.ravel()]) if stl_verts is not None else coords.ravel()
c_min  = float(all_vals.min())
c_max  = float(all_vals.max())
scale  = max(c_max - c_min, 1e-9)

coords_norm = (coords - c_min) / scale * 2 - 1

if stl_verts is not None:
    stl_norm = ((stl_verts - c_min) / scale * 2 - 1).astype(np.float32)
    stl_flat = np.round(stl_norm, 4).ravel().tolist()
else:
    stl_flat = []

# Pressure range
p_min, p_max = float(pressure.min()), float(pressure.max())

# Mesh pressure colours [0,1] per vertex — for pressure-coloured solid mesh
if stl_verts is not None and show_mesh:
    try:
        mesh_pressure_norm = get_mesh_pressure_norm(snap, stride_ui, str(stl_path))
        mesh_colors_flat   = mesh_pressure_norm.tolist()
    except Exception:
        mesh_colors_flat = []
else:
    mesh_colors_flat = []

# Point-cloud colour values in [0, 1]
if color_mode == t("vr_colour_z"):
    z = coords_norm[:, 2]
    color_values = ((z - z.min()) / max(z.max() - z.min(), 1e-9)).tolist()
    legend_min   = f"{coords[:, 2].min():.4f} m"
    legend_max   = f"{coords[:, 2].max():.4f} m"
    legend_label = f"Z-height · {colorscale}"
elif color_mode == t("vr_colour_idx"):
    color_values = np.linspace(0.0, 1.0, n_pts).tolist()
    legend_min   = "first node"
    legend_max   = "last node"
    legend_label = f"Node index · {colorscale}"
else:
    dp = p_max - p_min
    color_values = ((pressure - p_min) / max(dp, 1e-9)).tolist()
    legend_min   = f"{p_min:.1f} Pa"
    legend_max   = f"{p_max:.1f} Pa"
    legend_label = f"Pressure (Pa) · {colorscale}"

# ── Build HTML ────────────────────────────────────────────────────────────────

payload = json.dumps({
    "positions":    coords_norm.flatten().tolist(),
    "colors":       color_values,
    "n":            n_pts,
    "legend_min":   legend_min,
    "legend_max":   legend_max,
    "color_mode":   color_mode,
    "snap":         snap,
    "stl":          stl_flat,
    "show_mesh":    show_mesh,
    "show_cloud":   show_cloud,
    "mesh_colors":  mesh_colors_flat,
    "mesh_opacity": opacity,
    "colorscale":   colorscale,
}, separators=(",", ":"))

bg_hex  = "#0e1117" if bg_dark else "#f8f9fa"
fg_hex  = "#e8e8e8" if bg_dark else "#111111"
pt_size = round(point_size / 120, 5)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:{bg_hex}; overflow:hidden; font-family:system-ui,sans-serif; }}
  canvas {{ display:block; }}

  #info {{
    position:absolute; top:10px; left:10px;
    color:{fg_hex}; font-size:13px; line-height:1.7;
    background:rgba(0,0,0,0.55); padding:10px 14px;
    border-radius:8px; max-width:310px; user-select:none;
  }}

  #fly-btn {{
    display:inline-block; margin-top:8px;
    background:rgba(80,180,255,0.18); border:1px solid rgba(80,180,255,0.55);
    color:{fg_hex}; padding:5px 16px; border-radius:6px;
    cursor:pointer; font-size:12px; transition:background 0.18s;
  }}
  #fly-btn:hover  {{ background:rgba(80,180,255,0.38); }}
  #fly-btn.active {{ background:rgba(255,110,60,0.28); border-color:rgba(255,110,60,0.7); }}

  #speed-hud {{ display:none; margin-top:5px; font-size:11px; opacity:0.72; }}

  #crosshair {{
    display:none; position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%); pointer-events:none;
    color:{fg_hex}; font-size:22px; opacity:0.55; line-height:1;
    text-shadow: 0 0 6px rgba(0,0,0,0.8);
  }}

  #legend {{
    position:absolute; bottom:70px; right:10px;
    color:{fg_hex}; font-size:12px;
    background:rgba(0,0,0,0.5); padding:10px 14px;
    border-radius:8px; pointer-events:none; width:190px;
  }}
  #legend-bar {{ width:100%; height:14px; border-radius:4px; margin:4px 0; }}
  #legend-labels {{ display:flex; justify-content:space-between; font-size:11px; }}

  #vr-wrap {{
    position:absolute; bottom:14px; left:50%;
    transform:translateX(-50%);
  }}
  #VRButton {{
    background:rgba(0,0,0,0.6) !important; border:1px solid {fg_hex} !important;
    color:{fg_hex} !important; border-radius:6px !important;
    padding:8px 20px !important; font-size:14px !important; cursor:pointer;
  }}
</style>
</head>
<body>

<div id="info">
  <span id="mode-label" style="font-weight:bold;font-size:14px;letter-spacing:.5px">ORBIT MODE</span><br>
  <span id="mode-hint">
    Snapshot <strong>{snap}</strong> &nbsp;·&nbsp; {n_pts:,} pts &nbsp;·&nbsp; {colorscale}<br>
    <small style="opacity:0.7">Left-drag: orbit &nbsp;·&nbsp; Scroll: zoom &nbsp;·&nbsp; Right-drag: pan</small>
  </span>
  <div><div id="fly-btn" onclick="toggleFly()">🚀 Fly Mode &nbsp;<kbd>F</kbd></div></div>
  <div id="speed-hud">Speed: <span id="speed-val">8</span> &nbsp;·&nbsp; scroll to adjust</div>
</div>

<div id="crosshair">+</div>

<div id="legend">
  <div style="font-size:11px;opacity:0.85">{legend_label}</div>
  <canvas id="legend-bar" width="200" height="14"></canvas>
  <div id="legend-labels">
    <span id="lbl-min"></span><span id="lbl-max"></span>
  </div>
</div>

<div id="vr-wrap"></div>

<script type="importmap">
{{
  "imports": {{
    "three":         "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ VRButton      }} from 'three/addons/webxr/VRButton.js';

const DATA = {payload};

// ── Colormaps ─────────────────────────────────────────────────────────────────
function _lerp(stops, t) {{
  const seg=stops.length-1, idx=Math.min(Math.floor(t*seg),seg-1), f=t*seg-idx;
  const a=stops[idx], b=stops[idx+1];
  return [a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2])];
}}
const SCALES = {{
  Jet:    [[0,0,0.5],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0],[0.5,0,0]],
  Viridis:[[0.267,0.005,0.329],[0.190,0.408,0.563],[0.128,0.566,0.551],[0.369,0.788,0.383],[0.993,0.906,0.144]],
  RdBu_r: [[0.647,0.0,0.149],[0.843,0.188,0.153],[0.957,0.427,0.263],[0.992,0.682,0.58],[1,0.96,0.941],[0.818,0.898,0.941],[0.502,0.718,0.855],[0.263,0.576,0.765],[0.129,0.4,0.675],[0.031,0.188,0.42]],
  Plasma: [[0.05,0.03,0.528],[0.459,0.044,0.635],[0.699,0.025,0.611],[0.882,0.185,0.441],[0.975,0.417,0.248],[0.993,0.682,0.132],[0.94,0.975,0.131]],
}};
function getColor(t, scale) {{
  return _lerp(SCALES[scale] || SCALES.Viridis, Math.max(0, Math.min(1, t)));
}}

// ── Renderer ──────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);
document.getElementById('vr-wrap').appendChild(VRButton.createButton(renderer));

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color('{bg_hex}');
scene.fog = new THREE.Fog('{bg_hex}', 2.5, 7.0);

// ── Lights ────────────────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.45));

const hemi = new THREE.HemisphereLight(0xfff4e0, 0x334466, 0.55);
hemi.position.set(0, 1, 0);
scene.add(hemi);

const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
dir1.position.set(2, 3, 4);
dir1.castShadow = true;
dir1.shadow.mapSize.set(1024, 1024);
scene.add(dir1);

const dir2 = new THREE.DirectionalLight(0xaaddff, 0.4);
dir2.position.set(-3, -1, -2);
scene.add(dir2);

// ── Solid mesh — pressure-coloured via selected colorscale ────────────────────
if (DATA.show_mesh && DATA.stl.length > 0) {{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(DATA.stl), 3));

  const nV = DATA.stl.length / 3;
  const colBuf = new Float32Array(nV * 3);
  const hasP   = DATA.mesh_colors && DATA.mesh_colors.length === nV;
  for (let i = 0; i < nV; i++) {{
    const val = hasP ? DATA.mesh_colors[i] : 0.5;
    const [r,g,b] = getColor(val, DATA.colorscale);
    colBuf[i*3]=r; colBuf[i*3+1]=g; colBuf[i*3+2]=b;
  }}
  geo.setAttribute('color', new THREE.BufferAttribute(colBuf, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshPhongMaterial({{
    vertexColors: true,
    specular:     0x222233,
    shininess:    55,
    side:         THREE.DoubleSide,
    transparent:  DATA.mesh_opacity < 0.999,
    opacity:      DATA.mesh_opacity,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow    = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
}}

// ── Pressure point cloud (optional overlay) ───────────────────────────────────
let mat_pts = null;
if (DATA.show_cloud) {{
  const positions = new Float32Array(DATA.positions);
  const colBuf    = new Float32Array(DATA.n * 3);
  for (let i = 0; i < DATA.n; i++) {{
    const [r,g,b] = getColor(DATA.colors[i], DATA.colorscale);
    colBuf[i*3]=r; colBuf[i*3+1]=g; colBuf[i*3+2]=b;
  }}
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colBuf, 3));
  mat_pts = new THREE.PointsMaterial({{
    size: {pt_size}, vertexColors: true, sizeAttenuation: true,
  }});
  scene.add(new THREE.Points(geo, mat_pts));
}}

// ── Colorscale legend (always visible) ───────────────────────────────────────
const bar  = document.getElementById('legend-bar');
const lctx = bar.getContext('2d');
const grad = lctx.createLinearGradient(0, 0, bar.width, 0);
for (let i = 0; i <= 20; i++) {{
  const t = i/20, [r,g,b] = getColor(t, DATA.colorscale);
  grad.addColorStop(t, `rgb(${{(r*255)|0}},${{(g*255)|0}},${{(b*255)|0}})`);
}}
lctx.fillStyle = grad;
lctx.fillRect(0, 0, bar.width, bar.height);
document.getElementById('lbl-min').textContent = DATA.legend_min;
document.getElementById('lbl-max').textContent = DATA.legend_max;

// ── Camera ────────────────────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(0, 0, 2.8);
camera.rotation.order = 'YXZ';

// ── Orbit controls ────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping      = true;
controls.dampingFactor      = 0.07;
controls.screenSpacePanning = true;

const axes = new THREE.AxesHelper(0.22);
axes.position.set(-1.05, -1.05, 0);
scene.add(axes);

// ── Fly mode state ────────────────────────────────────────────────────────────
let flyMode    = false;
let flySpeed   = 0.008;
let yaw        = 0;
let pitch      = 0;
let isDragging = false;
let lastX = 0, lastY = 0;
const SENSITIVITY = 0.003;
const keys = {{}};

const modeLabel = document.getElementById('mode-label');
const modeHint  = document.getElementById('mode-hint');
const flyBtn    = document.getElementById('fly-btn');
const speedHud  = document.getElementById('speed-hud');
const speedVal  = document.getElementById('speed-val');
const crosshair = document.getElementById('crosshair');

window.toggleFly = () => setFlyMode(!flyMode);

const PT_SIZE_BASE = {pt_size};

function setFlyMode(on) {{
  flyMode = on;
  controls.enabled = !on;

  if (on) {{
    camera.position.set(0, 0, 1.15);
    yaw   = 0;
    pitch = -1.1;
    camera.rotation.set(pitch, yaw, 0, 'YXZ');
    camera.fov = 78; camera.updateProjectionMatrix();
    if (mat_pts) {{ mat_pts.size = PT_SIZE_BASE * 2.2; mat_pts.needsUpdate = true; }}
    scene.fog.near = 0.5; scene.fog.far = 2.5;

    modeLabel.textContent   = '✈  FLY MODE';
    modeHint.innerHTML      = '<small><b>Left-drag</b>: look around &nbsp;·&nbsp; <b>WASD</b>: fly<br><b>Q/E</b>: up/down &nbsp;·&nbsp; <b>Scroll</b>: speed &nbsp;·&nbsp; <b>F</b>: exit</small>';
    flyBtn.textContent      = 'Exit  F';
    flyBtn.classList.add('active');
    speedHud.style.display  = 'block';
    crosshair.style.display = 'block';
  }} else {{
    camera.position.set(0, 0, 2.8);
    camera.fov = 60; camera.updateProjectionMatrix();
    if (mat_pts) {{ mat_pts.size = PT_SIZE_BASE; mat_pts.needsUpdate = true; }}
    scene.fog.near = 2.5; scene.fog.far = 7.0;
    controls.target.set(0, 0, 0);
    controls.update();

    modeLabel.textContent   = 'ORBIT MODE';
    modeHint.innerHTML      = 'Snapshot <strong>{snap}</strong> &nbsp;·&nbsp; {n_pts:,} pts &nbsp;·&nbsp; {colorscale}<br><small style="opacity:0.7">Left-drag: orbit &nbsp;·&nbsp; Scroll: zoom &nbsp;·&nbsp; Right-drag: pan</small>';
    flyBtn.textContent      = '🚀 Fly Mode  F';
    flyBtn.classList.remove('active');
    speedHud.style.display  = 'none';
    crosshair.style.display = 'none';
  }}
}}

// ── Mouse look (left-drag in fly mode) ────────────────────────────────────────
renderer.domElement.addEventListener('mousedown', e => {{
  if (flyMode && e.button === 0) {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; }}
}});
window.addEventListener('mouseup', () => {{ isDragging = false; }});
window.addEventListener('mousemove', e => {{
  if (!flyMode || !isDragging) return;
  yaw   -= (e.clientX - lastX) * SENSITIVITY;
  pitch -= (e.clientY - lastY) * SENSITIVITY;
  pitch  = Math.max(-Math.PI/2 + 0.04, Math.min(Math.PI/2 - 0.04, pitch));
  lastX  = e.clientX; lastY = e.clientY;
  camera.rotation.set(pitch, yaw, 0, 'YXZ');
}});

// ── Keyboard ──────────────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {{
  keys[e.code] = true;
  if (e.code === 'KeyF') setFlyMode(!flyMode);
  if (flyMode && ['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code))
    e.preventDefault();
}});
window.addEventListener('keyup', e => {{ keys[e.code] = false; }});

renderer.domElement.addEventListener('wheel', e => {{
  if (!flyMode) return;
  flySpeed = Math.max(0.001, Math.min(0.06, flySpeed * (e.deltaY > 0 ? 0.88 : 1.14)));
  speedVal.textContent = Math.round(flySpeed * 1000);
  e.preventDefault();
}}, {{ passive: false }});

// ── Movement ──────────────────────────────────────────────────────────────────
const _fwd    = new THREE.Vector3();
const _right  = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);

function processFly() {{
  if (!flyMode) return;
  _fwd.set(0, 0, -1).applyEuler(camera.rotation).normalize();
  _right.set(1, 0, 0).applyEuler(camera.rotation).normalize();
  if (keys['KeyW'] || keys['ArrowUp'])    camera.position.addScaledVector(_fwd,      flySpeed);
  if (keys['KeyS'] || keys['ArrowDown'])  camera.position.addScaledVector(_fwd,     -flySpeed);
  if (keys['KeyA'] || keys['ArrowLeft'])  camera.position.addScaledVector(_right,   -flySpeed);
  if (keys['KeyD'] || keys['ArrowRight']) camera.position.addScaledVector(_right,    flySpeed);
  if (keys['KeyQ'] || keys['Space'])      camera.position.addScaledVector(_worldUp,  flySpeed);
  if (keys['KeyE'] || keys['ShiftLeft'])  camera.position.addScaledVector(_worldUp, -flySpeed);
}}

// ── Resize ────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Render loop ───────────────────────────────────────────────────────────────
renderer.setAnimationLoop(() => {{
  if (!renderer.xr.isPresenting) {{
    processFly();
    if (!flyMode) controls.update();
  }}
  renderer.render(scene, camera);
}});
</script>
</body>
</html>"""

# ── Embed in Streamlit ────────────────────────────────────────────────────────

components.html(html, height=660, scrolling=False)

# ── Download standalone file ──────────────────────────────────────────────────

col_dl, col_info = st.columns([1, 3])

with col_dl:
    st.download_button(
        label="⬇️ Download standalone HTML",
        data=html.encode("utf-8"),
        file_name=f"airway_vr_snap{snap}.html",
        mime="text/html",
        use_container_width=True,
        help=(
            "Open this file in Meta Quest Browser or any WebXR browser "
            "for immersive VR viewing — no server needed."
        ),
    )

with col_info:
    mesh_note = f"**{len(stl_flat)//9:,}** triangles" if stl_flat else "no STL found"
    st.caption(
        f"Mesh: {mesh_note} &nbsp;·&nbsp; "
        f"Cloud: **{n_pts:,}** nodes (every {stride_ui}th) &nbsp;·&nbsp; "
        f"Pressure: **{p_min:.1f}** → **{p_max:.1f}** Pa &nbsp;·&nbsp; "
        f"Colorscale: **{colorscale}**"
    )
