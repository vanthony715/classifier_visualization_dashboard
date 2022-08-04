# -*- coding: utf-8 -*-
"""
Creates Dashboard when Called from Main

@author: avasquez
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode,  plot
init_notebook_mode()

def createDash(TargetNames, PCA2D, TSNE2D, PCA3D, TSNE3D, ClassifierCM1, ClassifierCM2, DashWritePath):
    ## Make figure with subplots
    fig = make_subplots(rows=2, cols=3,
                        specs=[[{'type': 'xy'}, {'type': 'scene'}, {'type': 'heatmap'}],
                                        [{'type': 'xy'}, {'type': 'scene'}, {'type': 'heatmap'}]],
                        subplot_titles=("2D PCA", "3D PCA", "Random Forest Confusion Matrix",
                                        "2D t-SNE", "3D t-SNE", "KNN Confusion Matrix"))
    
    ## 2D PCA ##
    fig.add_trace(
        go.Scatter(x=PCA2D.E1, y=PCA2D.E2, customdata = PCA2D.Target,
                   name = '2D PCA',
                   hovertemplate='X: %{x:.2f}'+'<br>Y: %{y:.1f}' + '<br>Class: %{customdata}',
                    text=list(PCA2D['Target']), mode="markers", 
                            marker=dict(
                                        size=5,
                                        color=PCA2D['Target'],
                                        colorscale='dense',   # choose a colorscale
                                        opacity=0.7,
                                    )), 
        row = 1, col = 1)
            
    fig['layout']['xaxis1']['title']='Component 1'
    fig['layout']['yaxis1']['title']='Component 2'
    
    ## 3D PCA ##
    fig.add_trace(
        go.Scatter3d(x=PCA3D['E1'], y=PCA3D['E2'], z=PCA3D['E3'], opacity = 0.6,
                    name = '3D PCA',
                    customdata = PCA3D.Target,
                    hovertemplate='X: %{x:.1f}'+'<br>Y: %{y:.1f}' + '<br>Z: %{z:.1f}' +'<br>Class: %{customdata}',
                    text=list(PCA3D['Target']), mode="markers", 
                    marker=dict(
                                size=5,
                                color=PCA3D['Target'],
                                colorscale='dense',   # choose a colorscale
                                opacity=0.6,
                            )), 
        row = 1, col = 2)
    
    ##Random Forest CM
    ## Heatmap ##
    fig.add_trace(
        go.Heatmap(
                x=TargetNames,
                y=TargetNames,
                z=ClassifierCM1,
                colorscale= "gray",
                name = 'RF CM',
                hovertemplate='Class: %{x}'+'<br>Class: %{y}',
                showscale=False,
            ), row = 1, col = 3)
    
    
    fig['layout']['xaxis4']['title']='Target Classes'
    fig['layout']['yaxis4']['title']='Target Classes'
    
    ## 2D TSNE##
    fig.add_trace(
        go.Scatter(x=TSNE2D['E1'], y=TSNE2D['E2'], customdata = TSNE2D.Target,
                   name = '2D TSNE',
                   hovertemplate='X: %{x:.1f}'+'<br>Y: %{y:.1f}' + '<br>Class: %{customdata}',
                   opacity = 0.6, text=list(TSNE2D['Target']), mode="markers", 
                   marker=dict(
                            size=5,
                            color=TSNE2D.Target,
                            colorscale='dense',   # choose a colorscale
                            opacity=0.6,
                        )), 
        row = 2, col = 1)
    
    fig['layout']['xaxis3']['title']='Embedding 1'
    fig['layout']['yaxis3']['title']='Embedding 2'
    
    
    ## 3D TSNE ##
    fig.add_trace(
        go.Scatter3d(x=TSNE3D['E1'], y=TSNE3D['E2'], z=TSNE3D['E3'], opacity = 0.6,
                    name = '3D TSNE',
                    customdata = TSNE3D.Target,
                    hovertemplate='X: %{x:.1f}'+'<br>Y: %{y:.1f}' + '<br>Z: %{z:.1f}' +'<br>Class: %{customdata}',
                    text=list(TSNE3D['Target']), mode="markers", 
                    marker=dict(
                                size=6,
                                color=TSNE3D['Target'],
                                colorscale='dense',   # choose a colorscale
                                opacity=0.7,
                            )), 
        row = 2, col = 2)
    
    ## Heatmap ##
    fig.add_trace(
        go.Heatmap(
                x=TargetNames,
                y=TargetNames,
                z=ClassifierCM2,
                showlegend=False,
                name = 'KNN CM',
                colorscale= "gray",
                hovertemplate='Class: %{x}'+'<br>Class: %{y}',
            ), row = 2, col = 3)
    
    fig['layout']['xaxis2']['title']='Target Classes'
    fig['layout']['yaxis2']['title']='Target Classes'
    
    
    # # Hide legend
    fig.update_layout(
        title_text="Feature Separation Dash",
        height=1200,
        width=1800,
        template="plotly_dark",
    )
    
    fig.update_layout(legend_orientation="h", 
             xaxis1_rangeslider_visible=True, xaxis1_rangeslider_thickness=0.1,
             xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.1)
             
    
    # Update 3D plots
    fig.update_layout(scene = dict(
                    xaxis_title='Component 1',
                    yaxis_title='Component 2',
                    zaxis_title='Component 3'))
    
    fig.update_layout(scene2 = dict(
                    xaxis_title='Embedding 1',
                    yaxis_title='Embedding 2',
                    zaxis_title='Embedding 3'))
                    
    
    fig.update_layout(hovermode='closest')
    
    fig.update_layout(hoverlabel_align="left")
    
    # plot(fig)
    
    fig.write_html(DashWritePath)
        
    
