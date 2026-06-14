import os
import base64
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

RESULTS_DIR = "results"

DATASET_TO_TIMESERIES = {
    "metr-la": os.path.join(RESULTS_DIR, "benchmark_timeseries_metr-la.csv"),
    "pems-bay": os.path.join(RESULTS_DIR, "benchmark_timeseries_pems-bay.csv"),
}

def load_timeseries(dataset: str) -> pd.DataFrame:
    path = DATASET_TO_TIMESERIES[dataset]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing timeseries CSV: {path}")
    df = pd.read_csv(path)
    return df

def fig_lines(df: pd.DataFrame, x, series: dict, title: str, ytitle: str):
    fig = go.Figure()
    for name, col in series.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=ytitle,
        template="plotly_white",
        height=320,
        margin=dict(l=30, r=15, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def fig_vline(fig, xval):
    fig.add_vline(x=xval, line_width=2, line_dash="dot", line_color="black")
    return fig

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def layout_image_block(dataset: str, batch: int):
    # If you created topology snapshots, this will show them
    img_path = os.path.join(RESULTS_DIR, f"layout_frontier_{dataset}_batch{batch:03d}.png")
    if not os.path.exists(img_path):
        return html.Div(
            [
                html.Div("No topology snapshot found for this batch.", style={"color": "#666"}),
                html.Code(img_path, style={"fontSize": "12px"})
            ],
            style={"border": "1px solid #ddd", "padding": "10px", "borderRadius": "6px"}
        )
    return html.Div(
        [
            html.Div(f"Frontier layout snapshot (batch {batch})", style={"fontWeight": "600"}),
            html.Img(src=encode_image(img_path), style={"width": "100%", "marginTop": "8px"}),
        ],
        style={"border": "1px solid #ddd", "padding": "10px", "borderRadius": "6px"}
    )

app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Adaptive Frontier Louvain Dashboard"

app.layout = html.Div(
    [
        # Header block (separate from plots)
        html.Div(
            [
                html.H2("Adaptive Frontier Louvain — Results Dashboard",
                        style={"margin": "0 0 10px 0"}),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Dataset", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="dataset",
                                    options=[{"label": k, "value": k} for k in DATASET_TO_TIMESERIES.keys()],
                                    value="metr-la",
                                    clearable=False,
                                ),
                            ],
                            style={"minWidth": "240px", "flex": "0 0 240px"},
                        ),

                        html.Div(
                            [
                                html.Label("Batch", style={"fontWeight": "600"}),
                                dcc.Slider(
                                    id="batch",
                                    min=0, max=199, step=1, value=0,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ],
                            style={"flex": "1", "paddingLeft": "18px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "alignItems": "center",
                    },
                ),
            ],
            style={
                "position": "sticky",
                "top": "0",
                "zIndex": "999",
                "background": "white",
                "padding": "14px 12px 12px 12px",
                "borderBottom": "1px solid #e5e5e5",
            },
        ),

        # Content area
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="fig_time", config={"displayModeBar": False}),
                    ],
                    style={"flex": "1", "minWidth": "520px"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="fig_speedup", config={"displayModeBar": False}),
                    ],
                    style={"flex": "1", "minWidth": "520px"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "paddingTop": "12px"},
        ),

        html.Div(
            [
                html.Div([dcc.Graph(id="fig_modularity", config={"displayModeBar": False})],
                         style={"flex": "1", "minWidth": "520px"}),
                html.Div([dcc.Graph(id="fig_frontier", config={"displayModeBar": False})],
                         style={"flex": "1", "minWidth": "520px"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
        ),

        html.Div(
            [
                html.Div([dcc.Graph(id="fig_comms", config={"displayModeBar": False})],
                         style={"flex": "1", "minWidth": "520px"}),
                html.Div([html.Div(id="snapshot_panel")],
                         style={"flex": "1", "minWidth": "520px"}),
            ],
            style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "paddingBottom": "18px"},
        ),
    ],
    style={
        "padding": "0",  # header handles padding now
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    },
)

# app.layout = html.Div(
#     [
#         html.H2("Adaptive Frontier Louvain — Results Dashboard"),

#         html.Div(
#             [
#                 html.Label("Dataset"),
#                 dcc.Dropdown(
#                     id="dataset",
#                     options=[{"label": k, "value": k} for k in DATASET_TO_TIMESERIES.keys()],
#                     value="metr-la",
#                     clearable=False,
#                     style={"width": "220px"},
#                 ),
#             ],
#             style={"display": "inline-block", "marginRight": "25px"},
#         ),

#         html.Div(
#             [
#                 html.Label("Batch"),
#                 dcc.Slider(
#                     id="batch",
#                     min=0, max=199, step=1, value=0,
#                     tooltip={"placement": "bottom", "always_visible": False},
#                 ),
#             ],
#             style={"display": "inline-block", "width": "65%"},
#         ),

#         html.Hr(),

#         html.Div(
#             [
#                 html.Div([dcc.Graph(id="fig_time")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#                 html.Div([dcc.Graph(id="fig_speedup")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#             ]
#         ),

#         html.Div(
#             [
#                 html.Div([dcc.Graph(id="fig_modularity")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#                 html.Div([dcc.Graph(id="fig_frontier")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#             ]
#         ),

#         html.Div(
#             [
#                 html.Div([dcc.Graph(id="fig_comms")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#                 html.Div([html.Div(id="snapshot_panel")], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
#             ]
#         ),
#     ],
#     style={"padding": "18px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
# )

@app.callback(
    Output("batch", "max"),
    Output("batch", "value"),
    Input("dataset", "value"),
)
def update_slider(dataset):
    df = load_timeseries(dataset)
    max_batch = int(df["batch"].max())
    return max_batch, 0

@app.callback(
    Output("fig_time", "figure"),
    Output("fig_speedup", "figure"),
    Output("fig_modularity", "figure"),
    Output("fig_frontier", "figure"),
    Output("fig_comms", "figure"),
    Output("snapshot_panel", "children"),
    Input("dataset", "value"),
    Input("batch", "value"),
)
def update_figs(dataset, batch):
    df = load_timeseries(dataset)

    # Time plot
    fig_time = fig_lines(
        df, "batch",
        {
            "Static": "time_static",
            "ND": "time_nd",
            "DF": "time_df",
            "AF": "time_af",
        },
        title="Runtime per batch",
        ytitle="seconds",
    )
    fig_time = fig_vline(fig_time, batch)

    # Speedup plot
    fig_speedup = fig_lines(
        df, "batch",
        {
            "ND vs Static": "speedup_nd",
            "DF vs Static": "speedup_df",
            "AF vs Static": "speedup_af",
        },
        title="Speedup vs Static",
        ytitle="×",
    )
    fig_speedup = fig_vline(fig_speedup, batch)

    # Modularity plot
    fig_mod = fig_lines(
        df, "batch",
        {
            "Static": "modularity_static",
            "ND": "modularity_nd",
            "DF": "modularity_df",
            "AF": "modularity_af",
        },
        title="Modularity over time",
        ytitle="Q",
    )
    fig_mod = fig_vline(fig_mod, batch)

    # Frontier plot
    # If frontier_frac_af exists, use it. Else derive from affected counts if possible.
    fig_frontier = go.Figure()
    if "frontier_frac_af" in df.columns:
        fig_frontier.add_trace(go.Scatter(x=df["batch"], y=df["frontier_frac_af"], mode="lines", name="AF frontier frac"))
    if "n_affected_af" in df.columns:
        fig_frontier.add_trace(go.Scatter(x=df["batch"], y=df["n_affected_af"], mode="lines", name="AF affected (count)", yaxis="y2"))
        fig_frontier.update_layout(yaxis2=dict(overlaying="y", side="right", title="count"))
    fig_frontier.update_layout(
        title="Frontier adaptation",
        xaxis_title="batch",
        yaxis_title="fraction",
        template="plotly_white",
        height=320,
        margin=dict(l=30, r=30, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig_frontier = fig_vline(fig_frontier, batch)

    # Communities plot
    fig_comms = fig_lines(
        df, "batch",
        {
            "Static": "communities_static",
            "ND": "communities_nd",
            "DF": "communities_df",
            "AF": "communities_af",
        },
        title="#Communities over time",
        ytitle="count",
    )
    fig_comms = fig_vline(fig_comms, batch)

    snapshot = layout_image_block(dataset, int(batch))

    return fig_time, fig_speedup, fig_mod, fig_frontier, fig_comms, snapshot

if __name__ == "__main__":
    app.run(debug=True, port=8050)