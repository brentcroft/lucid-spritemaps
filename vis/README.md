# Usage:

### Review the example in the **eb_12_v07_480x270_01b** directory.


Open a console in the vis directory and run one of the following commands as required.<br>



```
python build_new_graph_vis.py
```
_Summary_
1. Read configuration
2. **Create missing directories and files.**

Edit the file "vis.js" to restrict the target layers and indexes that will be inspected (empty means all).
<br>
<br>


```
python build_sprites.py
```
_Summary_
1. Read configuration
2. Open the graph model.
3. Iterate over specified (instances and) layers and indexes
4. Not including existing work in **sprites** and **sprites_consumed** sub-directories.
5. Apply Lucid over a sequence of thresholds obtaining images and loss values.
6. Detect visualization problems (e.g. recurring loss value implies grey image)
7. Record loss values and status in CSV file at each threshold.
8. **Save images to sprites directory** with filename encoding layer, index and threshold.

Review the images in the **sprites** directory; remove any grey ones and adjust the transforms for that layer.
<br>
<br>


```
python build_spritemaps.py
```
_Summary_
1. Read configuration
2. **Create missing spritesheet files** (for each instance) for each layer, for each threshold 
3. Iterate over specified (instances and) layers and indexes.
4. Detect images in **sprites** directory.
5. **Update corresponding spritesheet and save**.
6. **Move images to sprites_consumed directory**.
<br>
<br>

View **spritemaps.html** in a browser.  
        