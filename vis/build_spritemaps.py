"""
    Create/Update Spritemaps
    


"""
import sys
sys.path.append("../lib")

import graph_models
import vis
import properties
PROPS = properties.Properties( "config.properties" )

graph_root = PROPS["GRAPHS"]
graph_version = PROPS["VIS_GRAPH"]
"""



    Build Spritemaps
"""    
vis.build_spritemaps( 
    
    vis = { "steps": [ 0 ] }, 

    graph_version = graph_version, 
    
    # note dummy parameter "steps" to match informal interface for a model_loader
    model_loader=lambda steps: graph_models.model_for_version( version = graph_version )
) 




