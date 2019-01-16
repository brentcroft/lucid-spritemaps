"""
    Create/Update Sprite images
"""
import os
import sys
sys.path.append("../lib")

import graph_models
import vis
import properties
PROPS = properties.Properties( "config.properties" )

graph_root = PROPS["GRAPHS"]
graph_version = PROPS["VIS_GRAPH"]
"""



    Obtain a file path for a frozen graph instance
"""
def get_graph_path( root='.', version=None, steps=None, name='frozen_inference_graph.pb' ):
    return os.path.join( root, "{}_{}".format( version, steps ), name )
"""



    Build Sprites
"""    
vis.build_sprites( 

    vis = { "steps": [ 0 ] }, 

    graph_version = graph_version, 
    
    model_loader=lambda steps: graph_models.model_for_version( 
        version = graph_version, 
        path = get_graph_path( graph_root, graph_version, steps ) 
    )
) 


  