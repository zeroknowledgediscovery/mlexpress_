# Machine Learning To Unravel Cellular Circuitry

---


## Python Module:

```
mlx.py
```

### Main script

```
qNet.py
```

#### Usage:

```
usage: qNet.py [-h] [--response RESPONSE [RESPONSE ...]] [--file FILE]
               [--filex FILEX] [--ntree NUMTREE] [--cores CORES]
               [--sample SAMPLES] [--plot [PLOT_]] [--varimp [VARIMP]]
               [--balance [BALANCE]] [--samplefeatures [SAMPLECOL]]
               [--del DELETE [DELETE ...]]
               [--inconly INCLUDEONLY [INCLUDEONLY ...]]
               [--inc INCLUDE [INCLUDE ...]] [--verbose [VERBOSE]]
               [--treename TREENAME] [--zerodel ZERODEL [ZERODEL ...]]
               [--importance_threshold FEATURE_IMP_THRESHOLD]
               [--edgefile EDGEFILE] [--dotfile DOTFILE]

Example with non-optional arguments:./dec_tree_2.py --file data.dat --filex
data.dat --varimp True --response DDR1 --zerodel B --del CELL
--importance_threshold 0.5

optional arguments:
  -h, --help            show this help message and exit
  --response RESPONSE [RESPONSE ...]
                        Response Variable
  --file FILE           train datafile
  --filex FILEX         test datafile
  --ntree NUMTREE       Number of trees in rndom forest
  --cores CORES         Number of cores to use in rndom forest
  --sample SAMPLES      sample size for columns
  --plot [PLOT_]        Show plot
  --varimp [VARIMP]     Feature importance
  --balance [BALANCE]   Balance class frequency of reposnse variable
  --samplefeatures [SAMPLECOL]
                        Choose a random sample of features
  --del DELETE [DELETE ...]
                        Deleted features
  --inconly INCLUDEONLY [INCLUDEONLY ...]
                        Included features, only
  --inc INCLUDE [INCLUDE ...]
                        Included features
  --verbose [VERBOSE]   Verbose
  --treename TREENAME
  --zerodel ZERODEL [ZERODEL ...]
                        Delete rows where response is in zerodel
  --importance_threshold FEATURE_IMP_THRESHOLD
                        Feature importance threshold: default 0.2
  --edgefile EDGEFILE   edges filename
  --dotfile DOTFILE     dot filename

```

---

#### Example:


```
./qNet.py --file /home/ishanu/ZED/Research/mlexpress_/data/qdat11.dat --filex /home/ishanu/ZED/Research/mlexpress_/data/qdat11.dat  --varimp True --response DDR1 --zerodel B --del CELL --importance_threshold 0.24
```

```
./pycode/qNet.py --file ./data/Zseq_.dat --filex ./data/Zseq_.dat  --varimp True --response 100  --importance_threshold 0.28 --edgefile edgetest.txt --dotfile edgetest.dot
```
