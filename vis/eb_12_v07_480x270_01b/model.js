graph = {
    "layers":[
        {
            "adam":0.05, 
            "cols":8, 
            "depth":32, 
            "index":0, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/FusedBatchNorm", 
            "transform_id":0
        }, 
        {
            "adam":0.05, 
            "cols":8, 
            "depth":64, 
            "index":1, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":0
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":128, 
            "index":2, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":0
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":128, 
            "index":3, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":1
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":256, 
            "index":4, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":1
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":256, 
            "index":5, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":1
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":6, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":1
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":7, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":1
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":8, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":9, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":2
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":10, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":2
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":11, 
            "name":"FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BatchNorm/FusedBatchNorm", 
            "transform_id":2
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":12, 
            "name":"FeatureExtractor/MobilenetV1/MaxPool2d_0_2x2/MaxPool", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":13, 
            "name":"FeatureExtractor/MobilenetV1/MaxPool2d_1_2x2/MaxPool", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":14, 
            "name":"FeatureExtractor/MobilenetV1/MaxPool2d_2_2x2/MaxPool", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":15, 
            "name":"FeatureExtractor/MobilenetV1/MaxPool2d_3_2x2/MaxPool", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":16, 
            "name":"FeatureExtractor/MobilenetV1/MaxPool2d_4_2x2/MaxPool", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":17, 
            "name":"WeightSharedConvolutionalBoxPredictor_5/PredictionTower/conv2d_0/BatchNorm/feature_5/FusedBatchNorm", 
            "transform_id":3
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":18, 
            "name":"WeightSharedConvolutionalBoxPredictor_4/PredictionTower/conv2d_0/BatchNorm/feature_4/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":19, 
            "name":"WeightSharedConvolutionalBoxPredictor_3/PredictionTower/conv2d_0/BatchNorm/feature_3/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":20, 
            "name":"WeightSharedConvolutionalBoxPredictor_2/PredictionTower/conv2d_0/BatchNorm/feature_2/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":21, 
            "name":"WeightSharedConvolutionalBoxPredictor_1/PredictionTower/conv2d_0/BatchNorm/feature_1/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":512, 
            "index":22, 
            "name":"WeightSharedConvolutionalBoxPredictor/PredictionTower/conv2d_0/BatchNorm/feature_0/FusedBatchNorm", 
            "transform_id":4
        }, 
        {
            "adam":0.05, 
            "cols":16, 
            "depth":12, 
            "index":23, 
            "name":"concat_1", 
            "transform_id":4
        }
    ]
}
