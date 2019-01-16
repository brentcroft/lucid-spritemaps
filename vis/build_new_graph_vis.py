"""
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


"""
vis.build_new_graph_vis( 

    graph_version = graph_version,
    
    # note dummy parameter "steps" to match informal interface for a model_loader
    model_loader=lambda steps: graph_models.model_for_version( version = graph_version )    
) 


  