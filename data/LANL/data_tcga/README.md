# Quantized TCGA Gene Expression Profiles
---

Quantization used: 

```
2 6
```


# To use different quantization (say *1 3 5*) with this data, rerun from top directory:

```
./data/LANL/getQuantized.sh ./data/LANL/all_norm_expr.txt  ./data/LANL/all_tumor_expr.txt  "1 3 5"
```


# command to run for Qnet regeneration:

```
../../pycode/qNet.py --file ./all_norm_exprmatched2000.csv --filex ./all_norm_exprmatched2000.csv  --varimp True --response BRCA1  --importance_threshold 0.28 --edgefile RESULTS/edgetest.txt --dotfile RESULTS/edgetest.dot

```
