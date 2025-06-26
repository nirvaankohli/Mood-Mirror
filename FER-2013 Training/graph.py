import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os 
import sys


V1 = pd.read_csv(os.path.join(sys.path[0], 'V1_validation_accuracy.csv'))

V2 = pd.read_csv(os.path.join(sys.path[0], 'V2_validation_accuracy.csv'))

V3 = pd.read_csv(os.path.join(sys.path[0], 'V3_validation_accuracy.csv'))

V4 = pd.read_csv(os.path.join(sys.path[0], 'V4_validation_accuracy.csv'))

for df in (V1, V2, V3, V4):

    if 'epoch' not in df.columns:
    
        df['epoch'] = np.arange(1, len(df) + 1)

def percent_bin_average(df, epoch_col='epoch', acc_col='val_acc', bins=100):
    
    df = df.sort_values(epoch_col).copy()
    N = df[epoch_col].max()
    
    # map epochs to [0,100)
    
    df['pct'] = (df[epoch_col] - 1) / (N - 1) * 100
    df['bin'] = np.floor(df['pct']).astype(int).clip(0, bins-1)
    
    # mean per bin
    
    grp = df.groupby('bin')[acc_col].mean().reindex(range(bins))
    
    x = np.arange(bins)     # 0…99
    valid = ~grp.isna().to_numpy()
    xp    = x[valid]
    fp    = grp.to_numpy()[valid]
    
    if len(xp) >= 2:
        y = np.interp(x, xp, fp)
    elif len(xp) == 1:
        # only one bin had data
        y = np.full(bins, fp[0], dtype=float)
    else:
        # no data at all? fill zeros (or pick any default)
        y = np.zeros(bins, dtype=float)
    
    return pd.DataFrame({
        'percent_training': x + 1,  # label 1…100
        'val_acc':           y
    })

V1_pct = percent_bin_average(V1).assign(model='V1')
V2_pct = percent_bin_average(V2).assign(model='V2')
V3_pct = percent_bin_average(V3).assign(model='V3')
V4_pct = percent_bin_average(V3).assign(model='V4')

df_plot = pd.concat([V1_pct, V2_pct, V3_pct, V4_pct], ignore_index=True)

fig = px.line(
    df_plot,
    x='percent_training',
    y='val_acc',
    color='model',
    title='Validation Accuracy vs % of Training'
)
fig.update_layout(
    xaxis_title='% of Training Complete',
    yaxis_title='Validation Accuracy',
    legend_title='Model'
)
fig.show()