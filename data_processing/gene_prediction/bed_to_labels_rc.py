import random
import numpy as np
import argparse
from intervaltree import Interval, IntervalTree

def build_interval_tree(gff_file):
    interval_trees = {}
    with open(gff_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[3]) - 1
            end = int(parts[4])
            if chrom not in interval_trees:
                interval_trees[chrom] = IntervalTree()
            interval_trees[chrom][start:end] = 1
    return interval_trees

def read_bed(bed_file):
    sequences = []
    with open(bed_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            sequences.append((chrom, start, end))
    return sequences

def create_labels_with_interval_tree(sequences, interval_trees):
    labels = []
    for chrom, seq_start, seq_end in sequences:
        label = np.zeros(seq_end - seq_start).astype(np.uint8)
        if chrom in interval_trees:
            overlapping_intervals = interval_trees[chrom].overlap(seq_start, seq_end)
            for interval in overlapping_intervals:
                overlap_start = max(seq_start, interval.begin)
                overlap_end = min(seq_end, interval.end)
                label[overlap_start-seq_start:overlap_end-seq_start] = 1
        label = [str(i) for i in label.tolist()]
        label = label[::-1]
        labels.append("".join(label))
    return labels

def save_labels(labels, output_file):
    with open(output_file, 'w') as f:
        for label in labels:
            f.write("".join(map(str, label)) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labels for sequences based on gene annotations from GFF file")
    parser.add_argument("-b", "--bed_file", type=str, required=True, help="Path to the input BED file containing sequence locations")
    parser.add_argument("-g", "--gff_file", type=str, required=True, help="Path to the GFF file containing annotations")
    parser.add_argument("-t", "--labels_file", type=str, required=True, help="Path to the output labels file")
    args = parser.parse_args()

    bed_file = args.bed_file
    gff_file = args.gff_file
    labels_file = args.labels_file

    sequences = read_bed(bed_file)
    interval_trees = build_interval_tree(gff_file)
    labels = create_labels_with_interval_tree(sequences, interval_trees)
    save_labels(labels, labels_file)

    print(f"Labels have been saved to {labels_file}")