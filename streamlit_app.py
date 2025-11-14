import streamlit as st
import cv2
import numpy as np
import base64
import io
from PIL import Image
import time
import json
import os

# Optional heavy imports — import lazily to keep the module importable in
# lightweight test environments where optional deps aren't present.
try:
    import potrace
except Exception:
    potrace = None

try:
    import svgwrite
except Exception:
    svgwrite = None

try:
    from scour import scour
except Exception:
    scour = None

# Optional metrics import (may require additional dependencies)
try:
    from metrics import calculate_all_metrics, rasterize_svg_to_edges
except Exception:
    calculate_all_metrics = None
    rasterize_svg_to_edges = None

# Optional visualization import
try:
    from visualization import (
        create_svg_overlay,
        create_comparison_overlay,
        create_edge_difference_map,
        create_metrics_visualization,
        create_side_by_side_comparison,
    )
except Exception:
    create_svg_overlay = None
    create_comparison_overlay = None
    create_edge_difference_map = None
    create_metrics_visualization = None
    create_side_by_side_comparison = None

# Optional validation metadata loader
try:
    from validation import load_test_metadata
except Exception:
    load_test_metadata = None

# Optional parameter optimizer
try:
    from parameter_optimizer import run_grid_search, GridSearchConfig, create_optimization_heatmap
except Exception:
    run_grid_search = None
    GridSearchConfig = None
    create_optimization_heatmap = None

# Utilities for validation and download
try:
    from utils import validate_image_array, is_blank_image
except Exception:
    validate_image_array = None
    is_blank_image = None

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")
st.title("Image to Engineering Drawing: Local Test App")

# --- SVG Display Function ---
def render_svg_display(svg_string: str):
    """
    Renders the given SVG string in Streamlit by embedding it as Base64.
   
    """
    try:
        b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
        # Add a border for better visibility
        html = r'<img src="data:image/svg+xml;base64,%s" style="border:1px solid #ddd; max-width: 100%%;"/>' % b64
        st.html(html)
    except Exception as e:
        st.error(f"Error rendering SVG: {e}")

# --- Core Processing Pipeline (Adapted from Serverless Report) ---

def process_image(image_array, canny_sigma, min_contour_frac):
    """
    Full image processing pipeline:
    1. Pre-process (Grayscale, Blur)
    2. Auto-Canny Edge Detection
    3. Contour Filtering (to remove noise)
    4. Potrace Vectorization
    """
    # Input validation
    # Raise explicit errors for invalid inputs to help callers handle them
    if validate_image_array is not None:
        validate_image_array(image_array)
    else:
        # Fallback validation when `utils.validate_image_array` is unavailable.
        # Mirror the key checks from utils.validate_image_array so behavior is
        # consistent between environments.
        if image_array is None:
            raise ValueError('image_array is None')

        if not hasattr(image_array, 'shape'):
            raise TypeError('image_array must be a numpy array-like with .shape')

        try:
            dims = image_array.shape
        except Exception:
            raise TypeError('Could not access image_array.shape')

        if len(dims) not in (2, 3):
            raise ValueError(f'Invalid image dimensions: expected 2 or 3, got {len(dims)}')

        h, w = int(dims[0]), int(dims[1])
        if h <= 0 or w <= 0:
            raise ValueError('Image has non-positive dimensions')
        if h < 10 or w < 10:
            raise ValueError('Image too small: minimum dimension is 10px')
        if h > 10000 or w > 10000:
            raise ValueError('Image too large: maximum dimension is 10000px')

        # dtype validation
        import numpy as _np
        if not _np.issubdtype(image_array.dtype, _np.integer) and not _np.issubdtype(image_array.dtype, _np.floating):
            raise TypeError(f'Image dtype must be numeric, got {image_array.dtype}')

        # NaN/Inf checks
        try:
            if _np.isnan(image_array).any() or _np.isinf(image_array).any():
                raise ValueError('Image array contains NaN or Inf values')
        except Exception:
            # If checks fail due to dtype, let earlier dtype error surface
            pass

    # 1. Pre-process
    # Ensure image is 3-channel BGR for consistent processing
    try:
        if len(image_array.shape) == 2:
            gray = image_array
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise TypeError(f'Failed to convert image to grayscale: {e}')

    # Blank/solid color detection
    try:
        if is_blank_image is not None and is_blank_image(image_array):
            # No useful edges to extract
            return None
        # fallback: check variance
        if is_blank_image is None and float(gray.std()) < 1.0:
            return None
    except Exception:
        # proceed if blank detection fails
        pass

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Auto-Canny
    # Compute median and apply automatic thresholds
    try:
        v = np.median(blurred)
        lower = int(max(0, (1.0 - canny_sigma) * v))
        upper = int(min(255, (1.0 + canny_sigma) * v))
        canny_edges = cv2.Canny(blurred, lower, upper)
    except Exception as e:
        # Numerical or OpenCV-related failure
        raise RuntimeError(f'Edge detection failed: {e}')

    # 3. Contour Filtering
    # Find all contours from the Canny output
    try:
        contours, _ = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except Exception as e:
        raise RuntimeError(f'Contour extraction failed: {e}')

    # Filter contours by area to remove noise
    image_area = gray.shape[0] * gray.shape[1]
    min_area = image_area * min_contour_frac

    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not significant_contours:
        return None # No significant edges detected

    # Create a new, clean mask by drawing *only* the significant contours
    clean_mask = np.zeros_like(canny_edges)
    cv2.drawContours(clean_mask, significant_contours, -1, (255), thickness=cv2.FILLED)

    # 4. Potrace Vectorization
    # Invert the mask (Potrace traces black areas)
    inverted_mask = (255 - clean_mask)
    
    # Create a Potrace bitmap from the NumPy array
    bmp = potrace.Bitmap(inverted_mask.astype(np.bool_))
    
    # Trace the bitmap to get path data
    path_data = bmp.trace()
    
    return path_data

def generate_svg_from_paths(path_data, width, height):
    """
    Uses svgwrite to convert Potrace path data into an SVG string.
    """
    dwg = svgwrite.Drawing(size=(width, height), profile='tiny')
    
    for curve in path_data:
        start = curve.start_point
        path_d = f"M{start.x},{start.y}"
        
        for segment in curve:
            if segment.is_corner:
                b = segment.c
                c = segment.end_point
                path_d += f" L{b.x},{b.y} L{c.x},{c.y}"
            else:
                b1 = segment.c1
                b2 = segment.c2
                c = segment.end_point
                path_d += f" C{b1.x},{b1.y} {b2.x},{b2.y} {c.x},{c.y}"

        dwg.add(dwg.path(
            d=path_d,
            fill="none",
            stroke="black",
            stroke_width=1
        ))

    return dwg.tostring()

def optimize_svg_string(svg_string):
    """
    Uses Scour to optimize/minify the SVG string in memory.
    """
    options = scour.sanitizeOptions(options=None)
    options.remove_metadata = True
    options.enable_id_stripping = True
    options.enable_comment_stripping = True
    options.shorten_ids = True
    options.indent_type = 'none' # Minify
    options.strip_xml_prolog = True
    options.remove_descriptive_elements = True
    
    optimized_svg = scour.scourString(svg_string, options)
    return optimized_svg

# --- Streamlit UI Elements ---

# Sidebar for controls
# Test case selector (optional)
if load_test_metadata is not None:
    try:
        metadata = load_test_metadata("test_images/test_metadata.json")
    except Exception:
        metadata = {}
else:
    metadata = {}

if 'selected_test_case' not in st.session_state:
    st.session_state['selected_test_case'] = None
    st.session_state['selected_optimal'] = None
    # Optimization session state
    st.session_state['optimization_results'] = None
    st.session_state['optimization_in_progress'] = False
    st.session_state['apply_optimized_params'] = False

with st.sidebar.expander("Load Test Case (optional)"):
    if metadata:
        options = [''] + list(metadata.keys())
        sel = st.selectbox("Select test case", options)
        if sel:
            st.session_state['selected_test_case'] = sel
            st.sidebar.write(f"Selected: {sel}")
            # Preload optimal params if available
            opt = metadata.get(sel, {}).get('optimal_parameters', {})
            if opt:
                st.session_state['selected_optimal'] = opt
    else:
        st.info("No test metadata found. Add test_images/test_metadata.json to enable test case selector.")

st.sidebar.header("Processing Parameters")

# Determine slider defaults: prefer applied optimized params, then selected_optimal from metadata
default_sigma = 0.33
default_min_contour = 0.0001
opt_res = st.session_state.get('optimization_results')
if st.session_state.get('apply_optimized_params') and opt_res and getattr(opt_res, 'best_params', None):
    bp = opt_res.best_params
    try:
        default_sigma = float(bp.canny_sigma)
        default_min_contour = float(bp.min_contour_frac)
    except Exception:
        pass
elif st.session_state.get('selected_optimal'):
    try:
        default_sigma = st.session_state['selected_optimal'].get('canny_sigma', default_sigma)
        default_min_contour = st.session_state['selected_optimal'].get('min_contour_frac', default_min_contour)
    except Exception:
        pass

canny_sigma = st.sidebar.slider(
    "Canny Sigma (Lower = more detail)", 
    min_value=0.0, max_value=1.0, 
    value=default_sigma, step=0.01
) #

min_contour_frac = st.sidebar.slider(
    "Noise Filter (Min Contour Area Fraction)",
    min_value=0.0, max_value=0.01, 
    value=default_min_contour, step=0.0001,
    format="%.4f" # Use printf style formatting
) #

st.sidebar.markdown("---")
st.sidebar.info("Upload a JPG or PNG file to begin processing.")

# Sidebar Quality Metrics (populated after processing completes)
st.sidebar.markdown("---")
st.sidebar.header("Quality Metrics")
metrics_state = st.session_state.get('metrics')
if metrics_state is not None:
    # metrics may be a dataclass/object or a dict (validation uses dict)
    def _m(k):
        if isinstance(metrics_state, dict):
            return metrics_state.get(k)
        return getattr(metrics_state, k, None)

    ep = _m('edge_precision')
    er = _m('edge_recall')
    ef = _m('edge_f1')
    pc = _m('path_count')
    comp = _m('compression_ratio_percent')
    hd = _m('hausdorff_distance')

    # Display metrics as cards; show N/A when not available
    try:
        if ep is not None:
            st.sidebar.metric("Edge Precision", f"{ep*100:.1f}%")
        else:
            st.sidebar.metric("Edge Precision", "N/A")

        if er is not None:
            st.sidebar.metric("Edge Recall", f"{er*100:.1f}%")
        else:
            st.sidebar.metric("Edge Recall", "N/A")

        if ef is not None:
            st.sidebar.metric("Edge F1", f"{ef*100:.1f}%")
        else:
            st.sidebar.metric("Edge F1", "N/A")

        if pc is not None:
            st.sidebar.metric("Path Count", f"{int(pc)}")
        else:
            st.sidebar.metric("Path Count", "N/A")

        if comp is not None:
            st.sidebar.metric("Compression", f"{comp:.1f}%")
        else:
            st.sidebar.metric("Compression", "N/A")

        if hd is not None:
            st.sidebar.metric("Hausdorff", f"{hd:.2f}")
        else:
            st.sidebar.metric("Hausdorff", "N/A")
    except Exception:
        # Defensive: if any formatting fails, show a compact summary
        st.sidebar.write(metrics_state)

# Visualization options
with st.sidebar.expander("Visualization Options"):
    show_overlay = st.checkbox("Show SVG Overlay on Original", value=False)
    show_comparison = st.checkbox("Show Comparison with Reference", value=False)
    show_diffmap = st.checkbox("Show Edge Difference Map", value=False)
    overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.7)

# Main panel for upload and display
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
) #

# Support either an uploaded file or a selected test-case from the metadata
image_array = None
image_height = image_width = None

if st.session_state.get('selected_test_case') and not uploaded_file:
    test_rel = st.session_state['selected_test_case']
    test_path = os.path.join('test_images', test_rel)
    if os.path.exists(test_path):
        file_bytes = np.asarray(bytearray(open(test_path, 'rb').read()), dtype=np.uint8)
        image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_array is not None:
            image_height, image_width = image_array.shape[:2]
    else:
        st.warning(f"Selected test image not found at {test_path}")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_height, image_width = image_array.shape[:2]

if image_array is None:
    st.info("Upload an image to begin processing.")
else:

    # --- Auto-Optimize Parameters (sidebar) ---
    with st.sidebar.expander("Auto-Optimize Parameters"):
        if run_grid_search is None:
            st.info("Parameter optimizer is not available in this environment.")
        else:
            st.write("Automatically search over canny_sigma and min_contour_frac to find good parameters for the current image.")
            search_type = st.selectbox("Search Strategy", ["quick", "standard", "fine"], index=0,
                                       help="Quick: 25 combinations, Standard: 100 combinations, Fine: 49 combinations around current values")
            auto_apply = st.checkbox("Auto-apply best parameters", value=True)

            # Use already-decoded image_array first; fallback to loading selected test case from disk
            def _load_image_for_opt_cached():
                # If image_array (decoded from uploader or previously loaded test case) is available, use it
                try:
                    if image_array is not None:
                        # return a copy to avoid accidental modification
                        return image_array.copy()
                except Exception:
                    pass

                # Fallback: load the selected test case from disk once
                if st.session_state.get('selected_test_case'):
                    test_rel = st.session_state['selected_test_case']
                    test_path = os.path.join('test_images', test_rel)
                    if os.path.exists(test_path):
                        try:
                            fb = open(test_path, 'rb').read()
                            file_bytes = np.asarray(bytearray(fb), dtype=np.uint8)
                            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        except Exception:
                            return None
                return None

            # Evaluate once and cache locally for this run
            _img_for_opt = _load_image_for_opt_cached()

            run_disabled = st.session_state.get('optimization_in_progress', False) or (_img_for_opt is None)
            if _img_for_opt is None:
                st.warning("Upload or select an image first to run optimization.")

            if st.button("Run Optimization", disabled=run_disabled):
                cfg = GridSearchConfig()
                cfg.search_type = search_type
                if search_type == 'fine':
                    cfg.center_point = (float(canny_sigma), float(min_contour_frac))

                st.session_state['optimization_in_progress'] = True
                progress_bar = st.progress(0)
                status_text = st.empty()

                def _progress_cb(frac, recent_result=None):
                    try:
                        progress_bar.progress(min(1.0, max(0.0, frac)))
                        if recent_result is not None:
                            status_text.text(f"Last: sigma={recent_result.canny_sigma:.3f}, frac={recent_result.min_contour_frac:.6g}, score={recent_result.score}")
                    except Exception:
                        pass

                try:
                    ref_svg = None
                    if st.session_state.get('selected_test_case') and metadata:
                        md = metadata.get(st.session_state['selected_test_case'], {})
                        expected_rel = md.get('expected_output')
                        if expected_rel and os.path.exists(expected_rel):
                            with open(expected_rel, 'r', encoding='utf-8') as f:
                                ref_svg = f.read()

                    if _img_for_opt is None:
                        raise RuntimeError('Could not load image for optimization')

                    opt_res = run_grid_search(_img_for_opt, reference_svg=ref_svg, config=cfg, progress_callback=_progress_cb)
                    st.session_state['optimization_results'] = opt_res
                    if auto_apply and getattr(opt_res, 'best_params', None):
                        bp = opt_res.best_params
                        st.session_state['selected_optimal'] = {'canny_sigma': bp.canny_sigma, 'min_contour_frac': bp.min_contour_frac}
                        st.session_state['apply_optimized_params'] = True
                    st.success('Optimization finished')
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                finally:
                    st.session_state['optimization_in_progress'] = False
                    progress_bar.empty()

            # show results if present
            if st.session_state.get('optimization_results'):
                orr = st.session_state['optimization_results']
                best = getattr(orr, 'best_params', None)
                if best:
                    st.metric('Best Score', f"{best.score:.2f}")
                    st.write(f"Best params: sigma={best.canny_sigma:.4f}, contour_frac={best.min_contour_frac:.6g}")
                    if st.button('Apply These Parameters'):
                        st.session_state['selected_optimal'] = {'canny_sigma': best.canny_sigma, 'min_contour_frac': best.min_contour_frac}
                        st.session_state['apply_optimized_params'] = True
                        st.experimental_rerun()

                with st.expander('Optimization Details'):
                    try:
                        tabs = st.tabs(["Summary", "Heatmap", "All Results"])
                        with tabs[0]:
                            rows = []
                            for r in sorted(orr.all_results, key=lambda x: (x.score or -1), reverse=True)[:5]:
                                rows.append({'sigma': r.canny_sigma, 'contour_frac': r.min_contour_frac, 'score': r.score, 'success': r.success})
                            st.dataframe(rows)
                        with tabs[1]:
                            try:
                                fig = create_optimization_heatmap(orr)
                                st.pyplot(fig)
                            except Exception as e:
                                st.write('Heatmap unavailable:', e)
                        with tabs[2]:
                            all_rows = [{'sigma': r.canny_sigma, 'contour_frac': r.min_contour_frac, 'score': r.score, 'success': r.success, 'time': r.processing_time} for r in orr.all_results]
                            st.dataframe(all_rows)
                    except Exception as e:
                        st.write('Could not display optimization details:', e)

    # Layout for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        else:
            # Display the selected test case loaded from disk
            st.image(image_array[:, :, ::-1], caption=st.session_state.get('selected_test_case'), channels='BGR', use_column_width=True)

    with col2:
        st.header("Generated SVG")

        # Run the full pipeline
        with st.spinner("Processing image..."):
            try:
                path_data = process_image(image_array, canny_sigma, min_contour_frac)

                if path_data:
                    # Generate and optimize SVG
                    svg_string = generate_svg_from_paths(path_data, image_width, image_height)
                    optimized_svg_string = optimize_svg_string(svg_string)

                    # Display the SVG
                    render_svg_display(optimized_svg_string)

                    # Add a download button for the SVG
                    download_name = None
                    if uploaded_file is not None:
                        download_name = uploaded_file.name.rsplit('.', 1)[0] + "_output.svg"
                    else:
                        download_name = (st.session_state.get('selected_test_case') or 'test').rsplit('.', 1)[0] + "_output.svg"

                    st.download_button(
                        label="Download Optimized SVG",
                        data=optimized_svg_string,
                        file_name=download_name,
                        mime="image/svg+xml"
                    )

                    # Compute metrics if reference available
                    metrics_obj = None
                    selected_ref = None
                    if st.session_state.get('selected_test_case') and metadata:
                        md = metadata.get(st.session_state['selected_test_case'], {})
                        expected_rel = md.get('expected_output')
                        if expected_rel and os.path.exists(expected_rel):
                            with open(expected_rel, 'r', encoding='utf-8') as f:
                                selected_ref = f.read()

                    if calculate_all_metrics is not None:
                        try:
                            metrics_obj = calculate_all_metrics(svg_string, optimized_svg_string, reference_svg=selected_ref, original_image_shape=image_array.shape)
                            st.session_state['metrics'] = metrics_obj
                        except Exception as e:
                            st.sidebar.warning(f"Metrics computation failed: {e}")

                    # Visualization tabs
                    tabs = st.tabs(["Original & SVG", "Overlay", "Comparison", "Difference Map", "Metrics Chart"])
                    with tabs[0]:
                        st.markdown("**Original** and **Generated SVG (rasterized)**")
                        # Prefer the helper which produces a side-by-side comparison when available
                        try:
                            if create_side_by_side_comparison is not None:
                                sb = create_side_by_side_comparison(image_array[:, :, ::-1], optimized_svg_string, selected_ref)
                                st.image(sb, caption=['Side-by-side Original and SVG'], use_column_width=True)
                            else:
                                # Fallback: show the original image only (SVG is shown in the Generated SVG panel)
                                st.image(image_array[:, :, ::-1], caption='Original', channels='BGR', use_column_width=True)
                        except Exception as e:
                            st.error(f"Side-by-side comparison failed: {e}")

                    with tabs[1]:
                        if show_overlay and create_svg_overlay is not None:
                            try:
                                overlay = create_svg_overlay(image_array[:, :, ::-1], optimized_svg_string, opacity=overlay_opacity)
                                st.image(overlay, caption='SVG Overlay on Original', use_column_width=True)
                            except Exception as e:
                                st.error(f"Overlay failed: {e}")
                        else:
                            st.info("Enable 'Show SVG Overlay on Original' in Visualization Options to view overlay.")

                    with tabs[2]:
                        if show_comparison and create_comparison_overlay is not None and selected_ref:
                            try:
                                comp = create_comparison_overlay(image_array[:, :, ::-1], optimized_svg_string, selected_ref, opacity=overlay_opacity)
                                st.image(comp, caption='Generated (green) vs Reference (red)', use_column_width=True)
                            except Exception as e:
                                st.error(f"Comparison overlay failed: {e}")
                        else:
                            st.info("Enable 'Show Comparison with Reference' and ensure a reference SVG is available.")

                    with tabs[3]:
                        # Guard on both rasterization and the difference-map helper being available
                        if show_diffmap and rasterize_svg_to_edges is not None and create_edge_difference_map is not None and selected_ref is not None:
                            try:
                                pred = rasterize_svg_to_edges(optimized_svg_string, image_width, image_height)
                                gt = rasterize_svg_to_edges(selected_ref, image_width, image_height)
                                diff = create_edge_difference_map(pred, gt)
                                st.image(diff, caption='Edge Difference Map (white=TP, red=FP, blue=FN)', use_column_width=True)
                            except Exception as e:
                                st.error(f"Difference map generation failed: {e}")
                        else:
                            st.info("Enable 'Show Edge Difference Map' and ensure reference SVG is available.")

                    with tabs[4]:
                        if metrics_obj is not None and create_metrics_visualization is not None:
                            try:
                                fig = create_metrics_visualization(metrics_obj)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Metrics visualization failed: {e}")
                        else:
                            st.info("Metrics are not available for this case.")

                    st.success("Conversion complete!")

                else:
                    st.warning("No significant edges were found with the current parameters. Try adjusting the sliders.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")