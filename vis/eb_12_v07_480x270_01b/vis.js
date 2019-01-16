vis = {
    "x-abort": true,
    
    "graph": "eb_12_v07_480x270_01b",
    
    "steps": [ "100k", "500k" ],
    
    "target_layers": [ 0, 1, 23 ],
    "target_indexes": [],
    
    "scale": 64,
    
    "thresholds": [ 64, 128, 256 ],
  
    "loss": {
        "num_bins": 10,
        "max_bin_hits": 10,
        "bin_factor": 10000000,
        "minimum_loss_threshold": 0
    }
}