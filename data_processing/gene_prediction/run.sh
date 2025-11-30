#!/bin/bash
set -e

# === 1. Receive Arguments ===
INPUT_FNA=$1
INPUT_GFF=$2

# Check if arguments are provided
if [[ -z "$INPUT_FNA" || -z "$INPUT_GFF" ]]; then
    echo "Error: Missing input files."
    echo "Usage: $0 <input.fna> <input.gff>"
    exit 1
fi

filename=$(basename "$INPUT_FNA")
SPECIES_NAME="${filename%.*}"

OUT_DIR="./$SPECIES_NAME"

# === 2. Create Output Directory ===
mkdir -p "$OUT_DIR"

# === 3. Copy and Process Files ===
# Note: Preserving original file logic but moving copies to the new directory
cp "$INPUT_FNA" "$OUT_DIR/$SPECIES_NAME.fna"
cp "$INPUT_GFF" "$OUT_DIR/$SPECIES_NAME.gff"

# Using variables $OUT_DIR and $SPECIES_NAME to replace hardcoded paths
# Extract regions and separate genes by strand (+/-)
awk '$3 == "region"' "$OUT_DIR/$SPECIES_NAME.gff" > "$OUT_DIR/regions.gff"
awk '$7 == "+" && $3 == "gene"' "$OUT_DIR/$SPECIES_NAME.gff" > "$OUT_DIR/pos_genes.gff"
awk '$7 == "-" && $3 == "gene"' "$OUT_DIR/$SPECIES_NAME.gff" > "$OUT_DIR/neg_genes.gff"

# === 4. Run Python Scripts ===
# Assuming python scripts are located in the current directory
python extract_full_sequence.py    -i "$OUT_DIR/$SPECIES_NAME.fna" -r "$OUT_DIR/regions.gff" -o "$OUT_DIR/pos.tsv" -b "$OUT_DIR/pos.bed" -l 65536
python extract_full_sequence_rc.py -i "$OUT_DIR/$SPECIES_NAME.fna" -r "$OUT_DIR/regions.gff" -o "$OUT_DIR/neg.tsv" -b "$OUT_DIR/neg.bed" -l 65536
python bed_to_labels.py    -b "$OUT_DIR/pos.bed" -g "$OUT_DIR/pos_genes.gff" -t "$OUT_DIR/pos_genes.tsv"
python bed_to_labels_rc.py -b "$OUT_DIR/neg.bed" -g "$OUT_DIR/neg_genes.gff" -t "$OUT_DIR/neg_genes.tsv"

# === 5. Merge Results ===
echo -e "sequence\tlabels" > "$OUT_DIR/$SPECIES_NAME.tsv"
paste "$OUT_DIR/pos.tsv" "$OUT_DIR/pos_genes.tsv" >> "$OUT_DIR/$SPECIES_NAME.tsv"
paste "$OUT_DIR/neg.tsv" "$OUT_DIR/neg_genes.tsv" >> "$OUT_DIR/$SPECIES_NAME.tsv"

echo "Done! Results saved in: $OUT_DIR/$SPECIES_NAME.tsv"