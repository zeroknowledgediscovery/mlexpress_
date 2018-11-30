forMarian2018Oct18.zip.  This zip file contains 4 matrix files:

breast_tumor_expr.txt
breast_norm_expr.txt
all_tumor_expr.txt
all_norm_expr.txt

Each is a matrix file with log2(TPM) expression values.  Rows labelled
by patient_id and columns by gene names.  Each file may have a
different number of genes since in each case a different number of
genes were removed due to having zero values for all samples.  The
starting number of genes before removal was 17,743.

Files with “tumor” in their name are for tumor samples.  Files with
“norm” are normal samples.  To distinguish variable names for paired
files (e.g. breast_tumor_expr.txt and breast_norm_expr.txt), gene
names for tumor files have “_t” added, while gene names in normal
files have “_n” added.

FYI, unlike what I told you earlier, the patient_ids do not include
any indication of tumor type.  However, should you need it, I can
supply metadata for each patient, including tumor type.

The matrix files starting with “all” include data from the 693 TCGA
patients for which we have paired normal and tumor samples.  The files
starting with breast are the subset representing 112 patients with
breast tumors.
