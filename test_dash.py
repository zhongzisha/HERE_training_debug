




import sys,os

import dash
import dash_deck
from dash import html
import pydeck
from pydeck.types import String
import numpy as np
import pandas as pd



import pydeck
import pandas as pd
backbone = sys.argv[1]

DATA_URL = f"ST_{backbone}_umap3d.csv" # 
# DATA_URL = "https://raw.githubusercontent.com/ajduberstein/geo_datasets/master/small_waterfall.csv"
df = pd.read_csv(DATA_URL)
print(df.head())
target = [df.x.mean(), df.y.mean(), df.z.mean()]

point_cloud_layer = pydeck.Layer(
    "PointCloudLayer",
    data=DATA_URL,
    get_position=["x", "y", "z"],
    get_color=["r", "g", "b"],
    get_normal=[0, 0, 15],
    auto_highlight=True,
    pickable=True,
    point_size=3,
)

view_state = pydeck.ViewState(target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=5.3)
view = pydeck.View(type="OrbitView", controller=True)

r = pydeck.Deck(point_cloud_layer, initial_view_state=view_state, views=[view],
    tooltip={
        'html': '<b>{slide_name}</b><br/><b>Spot: ({px}, {py})</b><br/><img src="data_HiDARE_PLIP_20240208/20240202v4_ST/PanCancer2GPUsFP/shared_attention_imagenetPLIP/split1_e95_h224_density_vis/feat_before_attention_feat/test/patch_images_64/{slide_name}/x{px}_y{py}.JPEG"/>',
        'style': {
            'color': 'white'
        }
    })
r.to_html(f"ST_{backbone}_umap3d.html", css_background_color="#add8e6")


# app = dash.Dash(__name__)

# app.layout = html.Div(
#     dash_deck.DeckGL(r.to_json(), id="deck-gl")
# )


# if __name__ == "__main__":
#     app.run_server(debug=True)



