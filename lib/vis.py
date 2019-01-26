"""
    

"""
import os
import math
from copy import deepcopy
import json
import time

import numpy as np
from PIL import Image
import pandas as pd

"""
"""
import tensorflow as tf
import lucid.optvis.render as render

from lucid.misc.io.serialize_array import _normalize_array as normalize_array

import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.transform as transform

import lucid_images
"""

"""
current_milli_time = lambda: int(round(time.time() * 1000))
current_timestamp = lambda: time.strftime( "%Y-%m-%d %H:%M:%S", time.gmtime() )
    
def create_blank_images( dir=None, scales=[ 64, 128, 256, 512 ]):
    for d in scales:
        img = Image.new( 'RGB', ( d, d ), ( 255, 0, 0, 0 ) )
        img.save( os.path.join( dir, 'blank_{}x{}.gif'.format( d, d ) ), 'GIF', transparency=0 )    


def check_abort( dirs = [] ):
    vf_abort = [
        vf[0]
        for vf in [
            ( vf, update_dict_from_json( json_path=vf ) )
            for vf in dirs
        ]
        if 'abort' in vf[1] and vf[1][ 'abort' ]
    ]
    return vf_abort      
        
    
def get_transforms( complexity=0 ):
    if complexity == 4:
        return [
            transform.pad( 32 ),
            transform.jitter( 64 ),
            transform.random_scale( [ n / 100. for n in range( 80, 120 ) ] ),
            transform.random_rotate( list( range( -10, 11 ) ) + 5 * [ 0 ] ),  
            #transform.random_rotate( list( range( -10, 10 ) ) + list( range( -5, 5 ) ) + 10 * list( range( -2, 2 ) ) ),
            transform.jitter( 8 )
        ]
        
    if complexity == 3:
        return [
            transform.pad( 16 ),
            transform.jitter( 16 ),
            transform.random_scale( [ 1 + ( i - 5 ) / 50. for i in range( 11 ) ] ),
            transform.random_rotate( list( range( -10, 11 ) ) + 5 * [ 0 ] ),                
            transform.jitter( 8 )
        ]
        
    if complexity == 2:
        return transform.standard_transforms         
  
    if complexity == 1:
        # no rotations
        return [
            transform.pad( 16 ),
            transform.jitter( 32 ),
            transform.random_scale( [ n / 100. for n in range( 80, 120 ) ] ),
            transform.jitter( 2 )
        ]        
        
    else:
        # no transforms
        return []


def get_pil_image( array ):
    img = Image.fromarray( normalize_array( array ) )
    img.load()
    return img


    
def loss_logger( log_file=None, item=None, threshold=None, loss=None, status=None ):
    
    header = None if os.path.isfile( log_file ) else [
            "batch_id",
            "timestamp",
            "scale", 
            "adam", 
            "transforms",
            "layer", 
            "index", 
            "threshold", 
            "loss", 
            "status" 
        ]
        
    body = [
        str( item['batch_id'] ),
        str( item['timestamp'] ), 
        str( item['scale'] ),  
        str( item['adam'] ), 
        str( item['transforms'] ),
        str( item['layer'] ), 
        str( item['index'] ), 
        str( threshold ), 
        str( loss ), 
        '"{}"'.format( status )
    ]
    
    with open( log_file, 'a' ) as f:  
        if header is not None:
            f.write( "{}\n".format( ", ".join( header ) ) )
        f.write( "{}\n".format( ", ".join( body ) ) )    

    
    
    
def get_visualizations_and_losses( 
        model, 
        objective_f, 
        param_f=None, 
        optimizer=None,
        transforms=None,
        threshold_start=0,
        thresholds=( 64, 128, 256, 512, 1024 ),
        visualization_index=None,
        visualization_layer=None,
        minimum_loss=0,
        num_bins=100,
        max_bin_hits=10,
        bin_factor=10000000,
        loss_logger=None ):
    
    with tf.Graph().as_default(), tf.Session() as sess:

        T = render.make_vis_T( model, objective_f, param_f, optimizer, transforms )
        
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        
        tf.global_variables_initializer().run()

        images = []

        try:
            # loss not changing much is indicator of failed image
            bin_losses = []
            bin_hits = 0
            
            for i in range( threshold_start, max( thresholds ) + 1 ):
            
                threshold_loss, _ = sess.run( [ loss, vis_op ] )
                
                if num_bins > 0:
                
                    bin_loss = int( bin_factor * threshold_loss )
                    
                    if bin_loss not in bin_losses:
                        # truncate and append
                        bin_losses = bin_losses[ -num_bins: ]
                        bin_losses.append( bin_loss )
                    else:
                        bin_hits = bin_hits + 1

                    if bin_hits > max_bin_hits:
                        print( "\nLOSS CRISIS: feature={}:{}; bin=[{}], hits=[{}], threshold={}, loss=[{}]; aborting image".format( 
                            visualization_layer, 
                            visualization_index, 
                            bin_loss,
                            bin_hits, 
                            i, 
                            threshold_loss,
                             ) )
                             
                        if loss_logger is not None:
                            loss_logger( threshold_loss, i, 'RECURS_{}_{}'.format( bin_hits, max_bin_hits ) )                             
                        
                        return [] 
                
                if minimum_loss > 0 and abs( threshold_loss ) < minimum_loss:
                    print( "\nLOSS ANXIETY ({}): layer={}, index={}, threshold={}, loss={}".format( 
                        minimum_loss, 
                        visualization_layer, 
                        visualization_index, 
                        i, 
                        threshold_loss ) )
                        
                    if loss_logger is not None:
                        loss_logger( threshold_loss, i, 'MIN_{}'.format( minimum_loss ) )
                        
                    return []
                    
                if i in thresholds:
                    vis = t_image.eval()
                    images.append( [ i, vis, float( threshold_loss ), visualization_index, visualization_layer ] )
                    
                    if loss_logger is not None:
                        loss_logger( threshold_loss, i, '' )                    
                    

        except KeyboardInterrupt as e:
            print( "Interrupted optimization at step {:d}.".format( i + 1 ) )
            raise e

        return images


def store_visualizations_and_losses( v_l_lists, output_dir=None, scale=64 ):
    for v_l_list in v_l_lists:
        for v_l in v_l_list:
            threshold = v_l[0]
            vis = v_l[1]
            # loss = round( v_l[2], 2 )
            index = v_l[3]
            layer = v_l[4]
            
            pil_img = get_pil_image( np.hstack( vis ) )
            
            if not os.path.isdir( output_dir ):
                os.mkdir( output_dir )
            
            try:
                image_path = get_image_path( output_dir, layer, index, threshold, scale )
                
                pil_img.save( image_path )
            
                print( "  saved image: {}".format( image_path ) )            
            
            except Exception as e:
                print( "Failed saving image: {}".format( image_path ) )
                raise e
            
            #
            pil_img.close()
            v_l[1] = None
            vis = None
            

    
"""
    Spritemaps
"""

def get_spritemap_path( output_dir, layer_index, threshold ):
    return os.path.join( output_dir, "l_{}_t_{}.png".format( layer_index, threshold ) )    


def get_layer_spritemaps( output_dir=None, layer_index=None, thresholds=None, size=None ):
    spritemaps = {}
    for threshold in thresholds:
        spritemap_path = get_spritemap_path( output_dir, layer_index, threshold )
        spritemap = None
        
        try:
            if os.path.isfile( spritemap_path ):
                spritemap = Image.open( spritemap_path )
                spritemap.load()
                spritemaps[ threshold ] = ( spritemap_path, spritemap )
                #print( "Opened existing spritemap: [{}]".format( spritemap_path ) )
        except Exception as e:
            print( "Error opening existing spritemap (will overwrite): [{}]; {}".format( spritemap_path, e ) )
            
        if spritemap is None:
            try:
                spritemap = Image.new( "RGB", size )
                spritemap.save( spritemap_path, "PNG" )
                spritemaps[ threshold ] = ( spritemap_path, spritemap )
                print( "Created new spritemap: [{}]".format( spritemap_path ) )
            except Exception as e:
                raise ValueError( "Error creating new spritemap: size={}, layer=[{}], threshold=[{}], path=[{}]".format( size, layer_index, threshold, spritemap_path ), e )

    return spritemaps

def add_sprite( spritemap=None, sprite=None, size=( 64, 64 ), cols=1, index=0 ):

    image_width, image_height = sprite.size
    
    tile_width, tile_height = size
    
    top = tile_height * math.floor( index / cols )
    left = tile_width * ( index % cols )
    
    #bottom = top + tile_height
    #right = left + tile_width

    if image_width < tile_width:
        left = left + int( ( tile_width - image_width ) / 2 )
        
    if image_height < tile_height:
        top = top + int( ( tile_height - image_height ) / 2 )
    
    #box = (left,top,right,bottom)
    box = ( left, top )
    box = [ int(i) for i in box ]

    try:
        spritemap.paste( sprite, box )
    except Exception as e:
        print( "Error pasting sprite: box=[{}]; {}".format( box, e ) )
        raise e

  
def get_image_path( output_dir, layer, index, threshold, scale ):
    return os.path.join( output_dir, "l_{}_i_{}_t_{}_s_{}.png".format( layer, index, threshold, scale ) )  
    
def extract_layer( s ):
    prefix='l_'
    suffix='_i_'    
    start = s.find( prefix ) + len( prefix )
    end = s.find( suffix, start )
    return int( s[ start:end ] )
    
def extract_index( s, layer_index=0 ):
    prefix='l_{}_i_'.format( layer_index )
    suffix='_t_'
    start = s.find( prefix ) + len( prefix )
    end = s.find( suffix, start )
    return int( s[ start:end ] )

def extract_threshold( s ):
    prefix='_t_'
    suffix='_s_'    
    start = s.find( prefix ) + len( prefix )
    end = s.find( suffix, start )
    return int( s[ start:end ] )

def extract_scale( s ):
    prefix='_s_'
    suffix='.png'    
    start = s.find( prefix ) + len( prefix )
    end = s.find( suffix, start )
    return int( s[ start:end ] )    
    
    
def extract_coords( f ):
    prefix = 'l_'
    layer_index_sep = '_i_'    
    index_threshold_sep = '_t_'
    threshold_scale_sep = '_s_'
    suffix='.png'
    
    layer_start = len( prefix )
    layer_end = f.find( layer_index_sep, layer_start )
    index_start = layer_end + len( layer_index_sep )
    index_end = f.find( index_threshold_sep, index_start )
    threshold_start = index_end + len( index_threshold_sep )
    threshold_end = f.find( threshold_scale_sep, threshold_start )
    scale_start = threshold_end + len( threshold_scale_sep )
    scale_end = f.find( suffix, scale_start )
    
    layer = int( f[ layer_start:layer_end ] )
    index = int( f[ index_start:index_end ] )
    threshold = int( f[ threshold_start:threshold_end ] )
    scale = int( f[ scale_start:scale_end ] )
    
    return ( layer, index, threshold, scale )


    
    
    
"""
    Not strict JSON, as needs to load in html
"""    
def store_as_json( json_path=None, json_prefix='vis = ', object={} ):
    json_text = json_prefix + json.dumps( object, sort_keys=True, separators=( ', ', ':' ), indent=4 )
    with open ( json_path, "w") as f:
        print( json_text, file=f )
        
    print( "Created new model json file: {}".format( json_path ) ) 
    
def load_from_json( json_path=None, json_prefix=' = ' ):

    if os.path.isfile( json_path ):
        with open( json_path, "r") as f:
            t = f.read()
            s = t.find( json_prefix )
            return json.loads( t[ s + len( json_prefix ):] )

    return None
    
def update_dict_from_json( json_path=None, json_prefix='vis = ', updatee=None ):

    if updatee is None:
        updatee = {}
        
    if os.path.isfile( json_path ):
        try:
            with open( json_path, "r") as f:
                t = f.read()
                s = t.find( json_prefix )
                updatee.update( json.loads( t[ s + len( json_prefix ):] ) )
                #print( "Updated dict from: {}".format( json_path ) )
                
        except Exception as e:
            raise ValueError( "Failed to update from: {}".format( json_path ), e )

    return updatee        


    
def get_existing_sprite_details( sprite_dirs=[], scale=64 ):
    existing_sprite_details = {}
    for sd in sprite_dirs:
        # only interested in ones at current scale
        for layer, index, threshold, _ in [ coords for coords in [ extract_coords( f ) for f in os.listdir( sd ) ] if coords[3] == scale ]:
            if layer in existing_sprite_details:
                layer_d = existing_sprite_details[ layer ]
            else:
                layer_d = {}
                existing_sprite_details[ layer ] = layer_d
                
            if index in layer_d:
                index_d = layer_d[ index ]
            else:
                index_d = {}
                layer_d[ index ] = index_d
                
            if threshold not in index_d:
                index_d[ threshold ] = threshold
    return existing_sprite_details

"""
    

"""
def get_next_batch_id( loss_log_path=None ):

    try:
        df = pd.read_csv( loss_log_path )
        
        last_batch_id = df[ 'batch_id' ].max()

        next_batch_id = last_batch_id + 1
        
        return next_batch_id
        
    except:
        next_batch_id = 0
        
    print( "\nCalculated next batch id: {}\n".format( next_batch_id ) )
        
    return next_batch_id

"""
"""
def get_graph_model( graph_version=None, model_filename='model.js', model_loader=None ):
    
    model_json_prefix = "graph = "
    model_json_path = os.path.join( graph_version, model_filename )

    # make sure there's a json file with the model layers
    if not os.path.isfile( model_json_path ):
    
        graph_model = { "layers": model_loader( 'dummy-steps' ).get_layers() }
    
        store_as_json( 
            json_path=model_json_path, 
            json_prefix=model_json_prefix, 
            object=graph_model ) 

        print( "Created new graph model JS file: {}".format( model_json_path ) )
    else:
        print( "Loaded graph model JS file: {}".format( model_json_path ) )
        
    
    return load_from_json( json_path=model_json_path, json_prefix=model_json_prefix )     





"""






""" 
def build_new_graph_vis( local_root='.', graph_version=None, model_loader=None  ):

    if graph_version is None:
        raise ValueError( "graph_version cannot be None" )
        
    if model_loader is None:
        raise ValueError( "model_loader cannot be None" )         
        
    graph_version_path = os.path.join( local_root, graph_version )
     
    if not os.path.isdir( graph_version_path ):
        os.mkdir( graph_version_path )        
    
    
    # create_blank_images( dir=graph_version_path )
    
    vis_path = os.path.join( graph_version_path, "vis.js" )
    
    if not os.path.isfile( vis_path ):
    
        # create a default vis file
        store_as_json( 
            json_path=vis_path,
            json_prefix="vis = ", 
            object={
            
            "x-abort": True,
            
            "graph": graph_version,
            
            "steps": [ ],
            
            "target_layers": [ ],
            "target_indexes": [ ],
            
            "scale": 64,
            
            "thresholds": [ 64, 256, 1024 ],
          
            "loss": {
                "num_bins": 10,
                "max_bin_hits": 10,
                "bin_factor": 10000000,
                "minimum_loss_threshold": 0
            }
        } ) 
    
    # if none exists , creates a model json file for customization
    get_graph_model( graph_version=graph_version, model_loader=model_loader )    
    
    
    print( "Prepared graph vis directory: {}".format( graph_version_path ) )
        


          
def build_sprites( local_root='.', graph_version=None, model_loader=None, vis=None, layers=None, vis_filename='vis.js' ):

    if graph_version is None:
        raise ValueError( "graph_version cannot be None" )
        
    if model_loader is None:
        raise ValueError( "model_loader cannot be None" )        

    graph_version_path = os.path.join( local_root, graph_version )
     
    if not os.path.isdir( graph_version_path ):
        raise ValueError( "No graph vis directory: {}".format( graph_version_path ) )

        
    if vis is None:
        vis = {}
      
    update_dict_from_json( json_path=os.path.join( graph_version_path, vis_filename ), updatee=vis )

    
    # 
    graph_steps = vis[ 'steps' ] if 'steps' in vis else []

    if len( graph_steps ) == 0:
        print( "no graph instances" )
        return
    
    for graph_step in graph_steps:

        graph_step_dir = os.path.join( graph_version_path, graph_step )
        image_dir = os.path.join( graph_step_dir, 'sprites' )
        image_consumed_dir = os.path.join( graph_step_dir, 'sprites_consumed' )
        image_scum_dir = os.path.join( graph_step_dir, 'sprites_scum' )
        sprite_map_dir = os.path.join( graph_step_dir, 'spritemaps' )

        log_path = os.path.join( graph_step_dir, 'losses.csv' ) 

        
        if not os.path.isdir( graph_step_dir ):
            os.mkdir( graph_step_dir )
        if not os.path.isdir( image_dir ):
            os.mkdir( image_dir )
        if not os.path.isdir( image_consumed_dir ):
            os.mkdir( image_consumed_dir )       
        if not os.path.isdir( sprite_map_dir ):
            os.mkdir( sprite_map_dir )        

        sprite_dirs = [
            d
            for d in [ image_dir, image_consumed_dir, image_scum_dir ]
            if os.path.isdir( d )
        ]
            
        
        # graph step specific config
        step_vis = deepcopy( vis )    

        update_dict_from_json( json_path=os.path.join( graph_step_dir, vis_filename ), updatee=step_vis )

        
        #
        max_index = step_vis[ 'max_index' ] if 'max_index' in step_vis else 2048
        scale = step_vis['scale'] if 'scale' in step_vis else 64
        thresholds = step_vis['thresholds'] if 'thresholds' in step_vis else [ 64 ]
        vis_loss = step_vis[ 'loss' ] if 'loss' in step_vis else {}

        
        batch_id = get_next_batch_id( loss_log_path=log_path ) 


        # drives off model json - as might be customised
        graph_model = get_graph_model( graph_version=graph_version, model_loader=model_loader )
            
        layers = graph_model['layers']
        
        
        # if not None and not empty then only build sprites for these layers/indexes
        if 'target_layers' in vis:
            target_layers = vis['target_layers']
            layers = [ 
                layer
                for layer in layers
                if target_layers is None or len( target_layers ) == 0 or layer['index'] in target_layers
            ]    
            
        target_indexes = [] if 'target_indexes' not in vis else vis['target_indexes']
        
        use_cppn = True if 'param' in vis and vis['param'] == 'cppn' else False
        
        # load existing sprite details
        existing_sprite_details = get_existing_sprite_details( sprite_dirs=sprite_dirs, scale=scale )  
            
        print( "\nBUILDING SPRITES: graph_version={} steps={}".format( graph_version, graph_step ) )    
        print( "   layers={}".format( [ layer['index'] for layer in layers ] ) ) 

        
        for layer in layers:

            layer_name = layer['name']
            layer_index = layer['index']
            
            adam = layer['adam']
            transform_id = layer['transform_id']
            
            model = None
            
            optimizer = tf.train.AdamOptimizer( adam )
            transforms = get_transforms( transform_id ) 
            
            #
            existing_layer_sprites = existing_sprite_details[ layer_index ] if layer_index in existing_sprite_details else []

            try:                
                print( "\nLAYER: {}\n".format( layer ) )
                
                num_processed = 0
                
                for index in range( 0, max_index ):
                
                    # check for abort in vis files
                    vf_abort = check_abort( dirs=[
                                os.path.join( graph_version_path, vis_filename ),
                                os.path.join( graph_step_dir, vis_filename )
                            ] )
                    
                    if len( vf_abort ) > 0:
                        print( "\nDetected abort in vis files: {}".format( vf_abort ) ) 
                        return                        

                        
                        
                    # check any target indexes
                    if not ( target_indexes is None or len( target_indexes ) == 0 or index in target_indexes ):
                        continue;

                    # 
                    existing_index_sprite_thresholds = existing_layer_sprites[ index ] if index in existing_layer_sprites else []

                    # calculate work to do
                    # do all thresholds already existing
                    thresholds_to_generate = [ t for t in thresholds if t not in existing_index_sprite_thresholds ]

                    if len( thresholds_to_generate ) == 0:
                        continue

                    # can start from an existing threshold
                    max_existing_threshold = max( existing_index_sprite_thresholds ) if len( existing_index_sprite_thresholds ) > 0 else None
                    
                    if max_existing_threshold is not None and max_existing_threshold <= min( thresholds_to_generate ):
                        
                        threshold_start = max_existing_threshold + 1
                        
                        img_path = [ 
                            ip
                            for ip in [
                                get_image_path( sd, layer_index, index, max_existing_threshold, scale ) 
                                for sd in sprite_dirs 
                            ]
                            if os.path.isfile( ip )
                        ][0]
                        
                        with Image.open( img_path ) as im:
                            im.load()
                            
                            # make array
                            im_1 = np.array( im )
                            
                        # add dummy batch dimension
                        im_2 = np.expand_dims( im_1, axis=0 )
                        
                        # reduce less than one
                        init_val = im_2.astype( np.float32 ) / 256
                        
                        param_f = lambda: lucid_images.image( scale, fft=False, init_val=init_val )
                        
                    elif use_cppn:
                        threshold_start = 0
                        adam = 0.00055
                        optimizer = tf.train.AdamOptimizer( adam )
                        param_f = lambda: param.cppn( scale )                    

                    else:
                        threshold_start = 0
                        param_f = lambda: param.image( scale, fft=True, decorrelate=True )

                        
                    # drop the model regularly
                    if num_processed % 100 == 0:
                        print( "Reloading model ..." )
                        model = None
                        num_processed = 0
                    
                    if model is None:
                        model = model_loader( graph_step )
                        model.load_graphdef()

                        
                    # start the feature
                    print( "\nFEATURE: {}:{}\n".format( layer['name'], index ) )

                    log_item = {
                        "batch_id": batch_id,
                        "timestamp": current_milli_time(),
                        "scale": scale,
                        "adam": adam,
                        "transforms": transform_id,
                        "layer": layer_index,
                        "index": index
                    }
                    
                    
                    
                    visualizations = []
                    
                    try:
                        visualization = get_visualizations_and_losses(
                            model, 
                            objectives.channel( layer_name, index ), 
                            transforms=transforms, 
                            param_f=param_f,
                            optimizer=optimizer,
                            threshold_start=threshold_start,
                            thresholds=thresholds,
                            visualization_index=index,
                            visualization_layer=layer_index,
                            
                            minimum_loss=vis_loss[ 'minimum_loss_threshold' ] if 'minimum_loss_threshold' in vis_loss else 0,
                            num_bins=vis_loss[ 'num_bins' ] if 'num_bins' in vis_loss else 0,
                            max_bin_hits=vis_loss[ 'max_bin_hits' ] if 'max_bin_hits' in vis_loss else 0,
                            bin_factor=vis_loss[ 'bin_factor' ] if 'bin_factor' in vis_loss else 0,
                            
                            loss_logger=lambda l,t,s: loss_logger( 
                                log_file=log_path, 
                                item=log_item, 
                                threshold=t, 
                                loss=l, 
                                status=s ) )

                        num_processed = num_processed + 1
                                
                        if len( visualization ) == 0:
                            continue
                        
                        # check losses
                        losses = [ v[ 2 ] for v in visualization ]
                    
                        print( "\nLOSSES: feature={}:{}; {}\n".format( layer_index, index, losses ) )
                                        
                        visualizations.append( visualization )
                        
                    finally:                    
                        if len( visualizations ) > 0:
                            store_visualizations_and_losses( visualizations, output_dir=image_dir, scale=scale )

            except ValueError as e:
                msg = "{}".format( e )
                if 'slice index' in msg and 'out of bounds' in msg:
                    print( "Closing layer: slice index out of bounds: {}".format( e ) )  
                else:
                    raise e

"""




"""
def build_spritemaps( local_root='.', graph_version=None, model_loader=None, vis=None, vis_filename='vis.js' ):
   
    if graph_version is None:
        raise ValueError( "graph_version cannot be None" )
        
    if model_loader is None:
        raise ValueError( "model_loader cannot be None" )        

    graph_version_path = os.path.join( local_root, graph_version )
     
    if not os.path.isdir( graph_version_path ):
        raise ValueError( "No graph vis directory: {}".format( graph_version_path ) )
       
    if vis is None:
        vis = {}
       
    update_dict_from_json( json_path=os.path.join( graph_version_path, vis_filename ), updatee=vis )

    graph_steps = vis['steps'] 
        
        
    print( "\nBUILDING SPRITEMAPS: graph_version={}".format( graph_version ) )    
    print( "   instances={}".format( graph_steps ) )   
    
        
    for graph_step in graph_steps:

        graph_step_dir = os.path.join( graph_version_path, graph_step )
        image_dir = os.path.join( graph_step_dir, 'sprites' )
        image_consumed_dir = os.path.join( graph_step_dir, 'sprites_consumed' )
        sprite_map_dir = os.path.join( graph_step_dir, 'spritemaps' )

        
        if not os.path.isdir( graph_step_dir ):
            print( "Invalid step directory: {}".format( graph_step_dir ) )
            continue
            
        if not os.path.isdir( image_dir ):
            print( "Invalid step sprites directory: {}".format( image_dir ) )
            continue
           
        
        all_files = os.listdir( image_dir )            
        
        if len( all_files ) == 0:
            print( "Step sprites directory is empty: {}".format( image_dir ) )
            continue
        
            
        if not os.path.isdir( image_consumed_dir ):
            os.mkdir( image_consumed_dir )       
        if not os.path.isdir( sprite_map_dir ):
            os.mkdir( sprite_map_dir )  

        
        # graph step specific config
        step_vis = deepcopy( vis )    

        update_dict_from_json( json_path=os.path.join( graph_step_dir, vis_filename ), updatee=step_vis )    
    
        scale = step_vis['scale'] if 'scale' in step_vis else 64
        thresholds = step_vis['thresholds'] if 'thresholds' in step_vis else [ 64 ]
        

        print( "Sprites to map: count=[{}], step=[{}]".format( len( all_files ), graph_step ) )  

        # drives off model json - as might be customised
        graph_model = get_graph_model( graph_version=graph_version, model_loader=model_loader )
            
        layers = graph_model['layers']
            
        
        # if not None and not empty then only build sprites for these layers/indexes
        if 'target_layers' in vis:
            target_layers = vis['target_layers']
            layers = [ 
                layer
                for layer in layers
                if target_layers is None or len( target_layers ) == 0 or layer['index'] in target_layers
            ]              
            
            

        # pull any available sprite images onto layer spritemap
        for layer in layers:

            # cols can't be greater than depth
            depth = layer['depth']
            cols = min( layer['cols'], depth )
            layer_index = layer['index']

            # TODO: what if depth doesn't divide evenly by cols
            rows = int( depth / cols )

            spritemap_size = ( scale * cols, scale * rows )

            spritemaps = get_layer_spritemaps( 
                output_dir=sprite_map_dir, 
                layer_index=layer_index, 
                thresholds=thresholds, 
                size=spritemap_size
            )    


            # get files for this layer
            layer_file_prefix = "l_{}_i_".format( layer_index )

            changed_spritemaps = set()

            try:
                files = [
                    ( extract_index( f, layer_index=layer_index ), extract_threshold( f ), f )
                    for f in all_files
                    if f.startswith( layer_file_prefix )
                ]

                files.sort( key=lambda f: ( f[0], f[1] ) )


                for index, threshold, f in files:

                    if threshold not in spritemaps:
                        # stray image
                        continue
                        
                        
                    spritemap_path, spritemap = spritemaps[ threshold ]

                    img_path = os.path.join( image_dir, f )

                    with Image.open( img_path ) as im:
                        im.load()

                        try:
                            add_sprite( 
                                spritemap=spritemap, 
                                sprite=im, 
                                size=( scale, scale ), 
                                cols=cols, 
                                index=index 
                            )
                        except Exception as e:
                            raise ValueError( "Error adding sprite: scale=[{}], cols=[{}], {}".format( scale, cols, e ), e )

                    changed_spritemaps.add( threshold )

                    if image_consumed_dir is not None:
                        # move sprite
                        os.rename( img_path,  os.path.join( image_consumed_dir, f ) )

            except Exception as e:
                raise e
            finally:
                for threshold in changed_spritemaps:
                    spritemap_path, spritemap = spritemaps[ threshold ]
                    spritemap.save( spritemap_path )
                    spritemap.close()
                 
              