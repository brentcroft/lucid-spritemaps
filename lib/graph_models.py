from lucid.modelzoo.vision_base import Model

thresholds = ( 64, 128, 256, 512, 1024 )
ppn_max_layer_index = 23

"""
    generate a layer definition at index i
"""
def _get_ppn_layer( i ):

    #CONV_2D = "Conv2D"
    #RELU_6 = "Relu6"
    FUSED_BATCH_NORM = "BatchNorm/FusedBatchNorm"
    
    
    POINTWISE = "pointwise"
    #DEPTHWISE = "depthwise"
    
    op = FUSED_BATCH_NORM
    plane = POINTWISE

    if i == 0:
        return ( 
            0, 
            32, 
            "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/{}".format( op ) )

    if i in range( 1, 12 ):
        if i == 1:
            depth = 64
        elif i in range( 2, 4 ):
            depth = 128
        elif i in range( 4, 6 ):
            depth = 256
        elif i in range( 6, 12 ):
            depth = 512
            
        return ( 
            i, 
            depth, 
            "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_{}_{}/{}".format( i, plane, op ) )
    
    if i in range( 12, 17 ):
        t = ( i - 12 )
        return ( 
            i, 
            512, 
            "FeatureExtractor/MobilenetV1/MaxPool2d_{}_2x2/MaxPool".format( t ) )
        
    if i in range( 17, 22 ):
        t = ( 22 - i )
        return ( 
            i, 
            512, 
            "WeightSharedConvolutionalBoxPredictor_{}/PredictionTower/conv2d_0/BatchNorm/feature_{}/FusedBatchNorm".format( t, t ) )

    if i == 22:
        t = 0
        return ( 
            i,
            512, 
            "WeightSharedConvolutionalBoxPredictor/PredictionTower/conv2d_0/BatchNorm/feature_{}/FusedBatchNorm".format( t ) )
    
    if i == 23:
        return ( 
            i, 
            12, 
            "concat_1" )
    
    raise ValueError( "Invalid PPN layer index: {}".format( i ) )
    
    
   
"""
    generate a list of layer definitions
"""
def _get_ppn_layers( cols=16, adam=0.05 ):
    layers = []
    for i in range( 0, ppn_max_layer_index + 1 ):
        layer_index, depth, name = _get_ppn_layer( i )
        
        layers.append( {
            'index': layer_index,
            'depth': depth,
            'name': name,
            'adam': adam,
            'cols': 8 if i in range( 0, 2 ) else cols,
            'transform_id': 0 if i in range( 0, 3 ) else 1 if i in range( 3, 8 ) else 2 if i in range( 9, 12 ) else 3 if i in range( 12, 18 ) else 4
        } )
    return layers        
       

class SSD_Mnet1_PPN( Model ):
    
    def __init__(self, image_shape=None, graph_path=None, labels_path=None ):        
        self.model_path = graph_path
        self.labels_path = labels_path

        self.image_shape = image_shape
        self.image_value_range = (-1, 1) 
        self.input_name = "Preprocessor/sub"        
        
        super().__init__()
    
    def get_layers( self, cols=16, adam=0.05 ):
        return _get_ppn_layers( cols=cols, adam=adam )       

        
def model_for_version( version=None, path=None ):

    if "320x180" in version:
        return SSD_Mnet1_PPN( graph_path=path, image_shape=[ 320, 180, 3 ] )

    if "480x270" in version:
        return SSD_Mnet1_PPN( graph_path=path, image_shape=[ 480, 270, 3 ] )
        
    if "720x405" in version:
        return SSD_Mnet1_PPN( graph_path=path, image_shape=[ 720, 405, 3 ] )
        
    raise ValueError( "No model for graph_version: {}".format( version ) )        