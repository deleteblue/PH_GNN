#!/usr/bin/env python3
"""
Interactive XAI Explorer for Toxin Protein Classification.

Two linked views:
  Left  — UMAP scatter of learned embeddings (click to select a protein)
  Right — Residue attention bar chart + py3Dmol 3D structure viewer
           (hover on bar chart highlights residue in 3D view)

Optional: "Send to ChimeraX" button colors residues by attention weight
          in a running ChimeraX session via its REST API.

Run:
    python xai_dash_app.py
Then access at http://localhost:8050
(Use SSH port-forwarding on the cluster: ssh -L 8050:localhost:8050 <host>)
"""

import os, sys, json, requests
import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px

# ── Load precomputed XAI data ─────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'xai_data.pt')

if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found. Run extract_xai_data.py first.")
    sys.exit(1)

print("Loading xai_data.pt...")
data = torch.load(DATA_PATH, map_location='cpu')

umap_xy          = data['umap_coords']          # [N, 2]
labels           = data['labels']               # [N] int
label_names      = data['label_names']          # ['Calcium', 'Potassium', 'Sodium']
predictions      = data['predictions']          # [N] int
confidences      = data['confidences']          # [N, num_classes] softmax probs
correct          = data['correct']              # [N] 0/1
protein_ids      = data['protein_ids']          # [N] str
pdb_paths        = data['pdb_paths']            # [N] str
taxa_list        = data['taxa']                 # [N] str
node_attentions  = data['node_attentions']      # list[np.array]
res_attentions   = data['residue_attentions']   # list[dict]
model_used       = data.get('model_used', '?')

N = len(protein_ids)
n_correct = int(correct.sum())
print(f"Loaded {N} proteins. Accuracy: {n_correct}/{N} ({n_correct/N*100:.1f}%). Model: {model_used}")

# ── Color palettes ─────────────────────────────────────────────────────────────
CLASS_COLORS = {'Calcium': '#e74c3c', 'Potassium': '#2ecc71', 'Sodium': '#3498db'}
TAXA_COLORS  = {
    'Snake': '#e67e22', 'Cone Snail': '#9b59b6', 'Spider': '#1abc9c',
    'Scorpion': '#f39c12', 'Sea Anemone': '#e91e63', 'Unknown': '#95a5a6',
}
CORRECT_COLORS = {'Correct': '#2ecc71', 'Wrong': '#e74c3c'}

# ── py3Dmol HTML template ─────────────────────────────────────────────────────
PY3DMOL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; background:#1a1a2e; }}
  #viewer {{ width:100%; height:400px; position:relative; }}
  #legend {{
    position:absolute; bottom:8px; left:8px;
    background:rgba(0,0,0,0.7); color:white;
    padding:6px 10px; border-radius:6px; font-size:11px;
    font-family: monospace;
  }}
</style>
</head>
<body>
<div id="viewer"></div>
<div id="legend">
  <b>{protein_id}</b> &nbsp; {label} &nbsp; | &nbsp; {taxa}<br>
  Color: low attn &#8594; blue &#8594; red &#8594; high attn
</div>
<script>
var pdbData = `{pdb_content}`;
var attentionMap = {attention_json};

var viewer = $3Dmol.createViewer($("#viewer"), {{backgroundColor:"#1a1a2e"}});
viewer.addModel(pdbData, "pdb");

// Color each residue by attention weight (blue=low, red=high)
viewer.setStyle({{}}, {{cartoon: {{color:"#4a4e69"}}}});

// Normalize attention to 0-1 range for coloring
var keys = Object.keys(attentionMap);
var vals = keys.map(function(k){{return attentionMap[k];}});
var minV = Math.min.apply(null, vals) || 0;
var maxV = Math.max.apply(null, vals) || 1;
if (maxV === minV) maxV = minV + 1;

function attnToColor(v) {{
  var t = (v - minV) / (maxV - minV);
  // blue -> cyan -> yellow -> red
  var r, g, b;
  if (t < 0.25) {{
    r = 0; g = Math.round(t*4*255); b = 255;
  }} else if (t < 0.5) {{
    r = 0; g = 255; b = Math.round((1-(t-0.25)*4)*255);
  }} else if (t < 0.75) {{
    r = Math.round((t-0.5)*4*255); g = 255; b = 0;
  }} else {{
    r = 255; g = Math.round((1-(t-0.75)*4)*255); b = 0;
  }}
  return 'rgb('+r+','+g+','+b+')';
}}

keys.forEach(function(k) {{
  var resi = parseInt(k.split("_")[0]);
  var color = attnToColor(attentionMap[k]);
  viewer.setStyle({{resi: resi}}, {{cartoon: {{color: color}}}});
}});

// Highlight hovered residue from Dash
window.highlightResidue = function(resi) {{
  viewer.setStyle({{resi: resi}}, {{cartoon: {{color:"white"}}, stick: {{color:"white"}}}});
  viewer.render();
}};

viewer.zoomTo();
viewer.render();

// Listen for residue highlight messages from parent frame
window.addEventListener("message", function(event) {{
  if (event.data && event.data.type === "highlight") {{
    var resi = event.data.resi;
    // Reset all, then highlight
    keys.forEach(function(k) {{
      var r = parseInt(k.split("_")[0]);
      var color = attnToColor(attentionMap[k]);
      viewer.setStyle({{resi: r}}, {{cartoon: {{color: color}}}});
    }});
    viewer.setStyle({{resi: resi}}, {{cartoon: {{color:"white", opacity:1}}, stick: {{color:"white"}}}});
    viewer.render();
  }}
}});
</script>
</body>
</html>
"""

def read_pdb(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception:
        return "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C"

def build_3dmol_html(idx):
    """Build py3Dmol HTML for protein at index idx."""
    pinfo_id  = protein_ids[idx]
    label     = label_names[labels[idx]]
    taxa      = taxa_list[idx]
    pdb_path  = pdb_paths[idx]
    res_attn  = res_attentions[idx]

    pdb_content = read_pdb(pdb_path)

    # Build attention JSON: keys = "resnum_resname", values = score
    attn_json = {f"{rnum}_{rname}": score
                 for (rnum, rname), score in res_attn.items()}

    html_str = PY3DMOL_TEMPLATE.format(
        protein_id=pinfo_id,
        label=label,
        taxa=taxa,
        pdb_content=pdb_content.replace('`', "'"),  # escape backticks
        attention_json=json.dumps(attn_json),
    )
    return html_str

def build_residue_bar(idx):
    """Build Plotly bar chart of per-residue attention for protein at idx."""
    res_attn = res_attentions[idx]
    if not res_attn:
        return go.Figure().add_annotation(text="No residue data", showarrow=False)

    sorted_res = sorted(res_attn.items(), key=lambda x: x[0][0])  # sort by resnum
    resnums = [f"{rnum} {rname}" for (rnum, rname), _ in sorted_res]
    scores  = [score for _, score in sorted_res]
    rnums_only = [rnum for (rnum, _), _ in sorted_res]

    # Normalize
    smin, smax = min(scores), max(scores)
    norm = [(s - smin) / (smax - smin + 1e-8) for s in scores]

    # Color by normalized attention (blue-to-red)
    colors = [f'rgb({int(v*255)}, {int((1-abs(v-0.5)*2)*200)}, {int((1-v)*255)})'
              for v in norm]

    fig = go.Figure(go.Bar(
        x=list(range(len(resnums))),
        y=scores,
        marker_color=colors,
        customdata=list(zip(resnums, scores, rnums_only)),
        hovertemplate="<b>%{customdata[0]}</b><br>Attention: %{customdata[1]:.4f}<extra></extra>",
        name="Attention",
    ))
    fig.update_layout(
        title=dict(text=f"Residue Attention — {protein_ids[idx]}", font_size=13),
        xaxis_title="Residue index",
        yaxis_title="Attention score",
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#0f0f23',
        font=dict(color='white'),
        margin=dict(l=40, r=10, t=40, b=30),
        height=220,
        showlegend=False,
    )
    return fig

def build_umap_scatter(color_by='class', filter_by='all'):
    """Build UMAP scatter figure colored by class, taxa, or correctness."""
    class_names_arr = [label_names[l] for l in labels]
    pred_names_arr  = [label_names[p] for p in predictions]
    correct_arr     = ['Correct' if c else 'Wrong' for c in correct]

    if color_by == 'class':
        color_arr = class_names_arr
        color_map = CLASS_COLORS
    elif color_by == 'taxa':
        color_arr = taxa_list
        color_map = TAXA_COLORS
    else:  # correctness
        color_arr = correct_arr
        color_map = CORRECT_COLORS

    hover_texts = [
        f"<b>{pid}</b><br>"
        f"True: <b>{cls}</b> &nbsp; Pred: <b style='color:{'#2ecc71' if c else '#e74c3c'}'>{pred}</b><br>"
        f"Confidence: {confidences[i][predictions[i]]:.1%}<br>"
        f"Taxa: {t}"
        for i, (pid, cls, pred, c, t) in enumerate(
            zip(protein_ids, class_names_arr, pred_names_arr, correct, taxa_list))
    ]

    # Apply filter
    if filter_by == 'correct':
        indices = [i for i in range(N) if correct[i]]
    elif filter_by == 'wrong':
        indices = [i for i in range(N) if not correct[i]]
    else:
        indices = list(range(N))

    categories = list(dict.fromkeys(color_arr[i] for i in indices))
    traces = []
    for cat in categories:
        mask = [i for i in indices if color_arr[i] == cat]
        color = color_map.get(cat, '#aaaaaa')
        # Wrong predictions: use X marker
        symbol = 'x' if (color_by == 'correctness' and cat == 'Wrong') else 'circle'
        traces.append(go.Scatter(
            x=umap_xy[mask, 0],
            y=umap_xy[mask, 1],
            mode='markers',
            name=cat,
            marker=dict(color=color, size=5 if cat != 'Wrong' else 7,
                        opacity=0.8, symbol=symbol,
                        line=dict(width=0.3, color='white')),
            text=[hover_texts[i] for i in mask],
            hoverinfo='text',
            customdata=mask,
        ))

    n_shown = len(indices)
    title_txt = (f"UMAP — Learned Embeddings  "
                 f"({n_shown} proteins shown, {n_correct}/{N} correct overall)")
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title_txt, font_size=12),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#0f0f23',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,0.4)', bordercolor='white', borderwidth=0.5),
        margin=dict(l=40, r=10, t=40, b=30),
        height=520,
        hovermode='closest',
    )
    return fig

# ── Dash app layout ───────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Toxin XAI Explorer"

DARK_BG   = '#0f0f23'
PANEL_BG  = '#1a1a2e'
TEXT_COL  = '#e0e0ff'
ACCENT    = '#7c3aed'

app.layout = html.Div(style={'backgroundColor': DARK_BG, 'minHeight': '100vh',
                              'fontFamily': 'Segoe UI, sans-serif', 'color': TEXT_COL}, children=[

    # ── Title bar ─────────────────────────────────────────────────────────────
    html.Div(style={'backgroundColor': PANEL_BG, 'padding': '12px 24px',
                    'borderBottom': f'1px solid {ACCENT}',
                    'display': 'flex', 'alignItems': 'center', 'gap': '20px'}, children=[
        html.H2("Toxin Protein XAI Explorer", style={'margin': 0, 'color': '#a78bfa'}),
        html.Span(f"Model: {model_used}", style={'color': '#6b7280', 'fontSize': '13px'}),
        html.Div(style={'marginLeft': 'auto', 'display': 'flex', 'gap': '20px',
                        'alignItems': 'center'}, children=[
            html.Div([
                html.Label("Color by:", style={'fontSize': '12px', 'color': '#9ca3af',
                                               'marginRight': '6px'}),
                dcc.RadioItems(
                    id='color-by',
                    options=[{'label': 'Class',       'value': 'class'},
                             {'label': 'Taxa',        'value': 'taxa'},
                             {'label': 'Correct/Wrong', 'value': 'correctness'}],
                    value='class',
                    inline=True,
                    inputStyle={'marginRight': '3px'},
                    labelStyle={'marginRight': '10px', 'color': TEXT_COL,
                                'fontSize': '13px'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Label("Show:", style={'fontSize': '12px', 'color': '#9ca3af',
                                           'marginRight': '6px'}),
                dcc.Dropdown(
                    id='filter-by',
                    options=[{'label': 'All proteins',      'value': 'all'},
                             {'label': 'Correct only',      'value': 'correct'},
                             {'label': 'Wrong predictions', 'value': 'wrong'}],
                    value='all',
                    clearable=False,
                    style={'width': '160px', 'fontSize': '12px',
                           'backgroundColor': PANEL_BG, 'color': '#111'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ]),
    ]),

    # ── Main 2-column layout ──────────────────────────────────────────────────
    html.Div(style={'display': 'flex', 'height': 'calc(100vh - 100px)'}, children=[

        # Left — UMAP
        html.Div(style={'flex': '1', 'padding': '12px', 'minWidth': 0}, children=[
            dcc.Graph(id='umap-plot', style={'height': '100%'},
                      config={'displayModeBar': False}),
        ]),

        # Right — detail panel
        html.Div(style={'width': '460px', 'padding': '12px',
                        'borderLeft': f'1px solid {ACCENT}',
                        'display': 'flex', 'flexDirection': 'column', 'gap': '8px'}, children=[

            # Protein header
            html.Div(id='protein-header',
                     style={'padding': '8px 12px', 'backgroundColor': PANEL_BG,
                            'borderRadius': '6px', 'fontSize': '13px'},
                     children="Click a protein on the UMAP to explore it."),

            # Residue attention bar chart
            dcc.Graph(id='attention-bar', config={'displayModeBar': False}),

            # 3D structure viewer (py3Dmol via iframe)
            html.Div(style={'flex': 1, 'position': 'relative'}, children=[
                html.Iframe(
                    id='structure-viewer',
                    style={'width': '100%', 'height': '400px', 'border': 'none',
                           'borderRadius': '6px'},
                    srcDoc="<html><body style='background:#1a1a2e;color:#aaa;"
                           "display:flex;align-items:center;justify-content:center;"
                           "height:100vh;margin:0;font-family:sans-serif'>"
                           "<p>Select a protein to view structure</p></body></html>",
                ),
                # Hidden JS to send highlight messages to iframe
                html.Div(id='highlight-msg', style={'display': 'none'}),
            ]),

            # ChimeraX integration
            html.Details(style={'fontSize': '12px', 'color': '#9ca3af',
                                 'marginTop': '4px'}, children=[
                html.Summary("ChimeraX REST Integration", style={'cursor': 'pointer',
                                                                   'color': '#a78bfa'}),
                html.Div(style={'marginTop': '8px', 'display': 'flex',
                                'gap': '8px', 'alignItems': 'center'}, children=[
                    dcc.Input(
                        id='chimerax-url',
                        value='http://localhost:41184',
                        placeholder='ChimeraX REST URL',
                        style={'flex': 1, 'padding': '4px 8px', 'fontSize': '12px',
                               'backgroundColor': PANEL_BG, 'color': TEXT_COL,
                               'border': f'1px solid {ACCENT}', 'borderRadius': '4px'},
                    ),
                    html.Button("Color in ChimeraX", id='chimerax-btn',
                                style={'padding': '4px 10px', 'fontSize': '12px',
                                       'backgroundColor': ACCENT, 'color': 'white',
                                       'border': 'none', 'borderRadius': '4px',
                                       'cursor': 'pointer'}),
                ]),
                html.Div(id='chimerax-status',
                         style={'marginTop': '4px', 'color': '#6b7280', 'fontSize': '11px'}),
                html.Div(style={'marginTop': '4px', 'color': '#6b7280',
                                'fontSize': '11px', 'lineHeight': '1.5'}, children=[
                    "Start ChimeraX with: ",
                    html.Code("chimerax --rest 41184",
                              style={'backgroundColor': '#111', 'padding': '1px 4px'}),
                ]),
            ]),
        ]),
    ]),

    # Hidden store for selected protein index
    dcc.Store(id='selected-idx', data=None),
    dcc.Store(id='hovered-resi', data=None),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('umap-plot', 'figure'),
    Input('color-by', 'value'),
    Input('filter-by', 'value'),
)
def update_umap(color_by, filter_by):
    return build_umap_scatter(color_by, filter_by)


@app.callback(
    Output('selected-idx', 'data'),
    Input('umap-plot', 'clickData'),
    State('color-by', 'value'),
    prevent_initial_call=True,
)
def store_selection(click_data, color_by):
    """Translate click on UMAP trace to original protein index."""
    if not click_data:
        return None
    point = click_data['points'][0]
    # customdata holds the original index within the dataset
    idx = point.get('customdata')
    if idx is None:
        return None
    return int(idx)


@app.callback(
    Output('protein-header', 'children'),
    Output('attention-bar', 'figure'),
    Output('structure-viewer', 'srcDoc'),
    Input('selected-idx', 'data'),
    prevent_initial_call=True,
)
def update_detail_panel(idx):
    if idx is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor='#1a1a2e', paper_bgcolor='#0f0f23',
            font=dict(color='white'), height=220,
        )
        placeholder_html = ("<html><body style='background:#1a1a2e;color:#aaa;"
                            "display:flex;align-items:center;justify-content:center;"
                            "height:100vh;margin:0;font-family:sans-serif'>"
                            "<p>Select a protein to view structure</p></body></html>")
        return "Click a protein on the UMAP to explore it.", empty_fig, placeholder_html

    pid        = protein_ids[idx]
    true_label = label_names[labels[idx]]
    pred_label = label_names[predictions[idx]]
    is_correct = bool(correct[idx])
    conf       = confidences[idx]           # [num_classes] softmax probs
    taxa       = taxa_list[idx]
    ra         = res_attentions[idx]
    n_res      = len(ra)
    n_atm      = len(node_attentions[idx]) if idx < len(node_attentions) else 0

    verdict_color  = '#2ecc71' if is_correct else '#e74c3c'
    verdict_text   = 'CORRECT' if is_correct else 'WRONG'

    # Confidence breakdown bars
    conf_bars = []
    for ci, cname in enumerate(label_names):
        pct = conf[ci] * 100
        bar_color = CLASS_COLORS.get(cname, '#aaa')
        is_true = (cname == true_label)
        is_pred = (cname == pred_label)
        conf_bars.append(html.Div(style={'display': 'flex', 'alignItems': 'center',
                                          'gap': '6px', 'marginBottom': '3px'}, children=[
            html.Span(cname, style={'width': '80px', 'fontSize': '11px',
                                    'color': '#9ca3af',
                                    'fontWeight': 'bold' if (is_true or is_pred) else 'normal'}),
            html.Div(style={'flex': 1, 'backgroundColor': '#111',
                            'borderRadius': '3px', 'height': '10px',
                            'position': 'relative'}, children=[
                html.Div(style={'width': f'{pct:.1f}%', 'height': '100%',
                                'backgroundColor': bar_color, 'borderRadius': '3px',
                                'opacity': '0.9' if (is_true or is_pred) else '0.4'}),
            ]),
            html.Span(f"{pct:.1f}%", style={'width': '38px', 'fontSize': '11px',
                                             'textAlign': 'right', 'color': '#9ca3af'}),
            html.Span('← true' if is_true and not is_pred else
                      '← pred' if is_pred and not is_true else
                      '← true & pred' if is_true and is_pred else '',
                      style={'fontSize': '10px', 'color': verdict_color, 'width': '70px'}),
        ]))

    header_children = html.Div(style={'display': 'flex', 'flexDirection': 'column',
                                       'gap': '6px'}, children=[
        html.Div(style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap',
                        'alignItems': 'center'}, children=[
            html.Div([html.Span("Protein: ", style={'color': '#6b7280'}), html.B(pid)]),
            html.Div([html.Span("True: ", style={'color': '#6b7280'}),
                      html.B(true_label, style={'color': CLASS_COLORS.get(true_label, 'white')})]),
            html.Div([html.Span("Pred: ", style={'color': '#6b7280'}),
                      html.B(pred_label, style={'color': CLASS_COLORS.get(pred_label, 'white')})]),
            html.Span(verdict_text, style={'color': verdict_color, 'fontWeight': 'bold',
                                           'fontSize': '12px',
                                           'border': f'1px solid {verdict_color}',
                                           'padding': '1px 6px', 'borderRadius': '4px'}),
            html.Div([html.Span("Taxa: ", style={'color': '#6b7280'}),
                      html.B(taxa, style={'color': TAXA_COLORS.get(taxa, 'white')})]),
            html.Span(f"{n_res} res / {n_atm} atoms",
                      style={'color': '#6b7280', 'fontSize': '11px'}),
        ]),
        html.Div(conf_bars),
    ])

    bar_fig  = build_residue_bar(idx)
    html_str = build_3dmol_html(idx)

    return header_children, bar_fig, html_str


@app.callback(
    Output('chimerax-status', 'children'),
    Input('chimerax-btn', 'n_clicks'),
    State('chimerax-url', 'value'),
    State('selected-idx', 'data'),
    prevent_initial_call=True,
)
def send_to_chimerax(n_clicks, rest_url, idx):
    """
    Send attention-weighted coloring commands to ChimeraX REST server.
    ChimeraX must be running with: chimerax --rest 41184
    """
    if not n_clicks or idx is None:
        return ""

    pdb_path = pdb_paths[idx]
    ra       = res_attentions[idx]

    if not ra:
        return "No residue attention data for this protein."

    # Normalize attention to 0-100 range for B-factor replacement
    vals = list(ra.values())
    vmin, vmax = min(vals), max(vals)
    rng = (vmax - vmin) if vmax != vmin else 1.0

    # Build ChimeraX commands:
    # 1. Open the PDB file
    # 2. Color by "by attribute" using rangecolor after setting bfactor via attribute file
    # Simpler: use rangecolor with residue selection loops
    cmds = [f"open {pdb_path}"]

    # Color residues in 3 buckets: low (blue), mid (white), high (red)
    low_res  = [str(rnum) for (rnum, _), v in ra.items() if (v - vmin) / rng < 0.33]
    mid_res  = [str(rnum) for (rnum, _), v in ra.items() if 0.33 <= (v - vmin) / rng < 0.67]
    high_res = [str(rnum) for (rnum, _), v in ra.items() if (v - vmin) / rng >= 0.67]

    if low_res:
        cmds.append(f"color /A:{','.join(low_res)} blue")
    if mid_res:
        cmds.append(f"color /A:{','.join(mid_res)} white")
    if high_res:
        cmds.append(f"color /A:{','.join(high_res)} red")

    full_cmd = "; ".join(cmds)

    try:
        rest_url = rest_url.rstrip('/')
        resp = requests.post(
            f"{rest_url}/run",
            params={'command': full_cmd},
            timeout=5,
        )
        if resp.status_code == 200:
            return f"Sent to ChimeraX: {protein_ids[idx]} ({len(cmds)-1} color groups)"
        else:
            return f"ChimeraX error {resp.status_code}: {resp.text[:80]}"
    except Exception as e:
        return f"Connection failed: {e}"


# ── Residue hover → highlight in 3D viewer via JS postMessage ─────────────────
# Inject a clientside callback that sends a postMessage to the iframe
app.clientside_callback(
    """
    function(hoverData) {
        if (!hoverData) return null;
        var pt = hoverData.points[0];
        if (!pt || !pt.customdata) return null;
        var resi = pt.customdata[2];   // rnums_only from customdata
        var iframe = document.getElementById('structure-viewer');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({type: 'highlight', resi: resi}, '*');
        }
        return resi;
    }
    """,
    Output('hovered-resi', 'data'),
    Input('attention-bar', 'hoverData'),
    prevent_initial_call=True,
)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print(f"\nStarting XAI Dash app on http://{args.host}:{args.port}")
    print("On the cluster, access via:  ssh -L 8050:localhost:8050 <hostname>")
    app.run(host=args.host, port=args.port, debug=args.debug)
