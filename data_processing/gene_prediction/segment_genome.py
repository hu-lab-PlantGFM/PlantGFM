import argparse
import random
import os
from Bio import SeqIO
from tqdm import tqdm

def process_genome(input_file, output_file):
    # === Configuration ===
    SEQ_LEN = 65536        # Fixed sequence length
    OVERLAP_MIN = 60       # Minimum overlap
    OVERLAP_MAX = 100      # Maximum overlap
    MAX_N_RATIO = 0.05     # Filter threshold for 'N' content (5%)

    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Count total records for the progress bar (optional, can be skipped for speed)
    total_records = sum(1 for _ in SeqIO.parse(input_file, "fasta"))
    
    with open(output_file, 'w') as f_out:
        # Read FASTA file using Biopython
        records = SeqIO.parse(input_file, "fasta")
        
        count_saved = 0
        
        for record in tqdm(records, total=total_records, desc="Processing Chromosomes"):
            # Convert sequence to string and uppercase
            seq = str(record.seq).upper()
            seq_length = len(seq)
            
            current_pos = 0
            
            # Sliding window segmentation
            while current_pos + SEQ_LEN <= seq_length:
                # 1. Slice the sequence
                subseq = seq[current_pos : current_pos + SEQ_LEN]
                
                # 2. Calculate N content
                n_count = subseq.count('N')
                n_ratio = n_count / SEQ_LEN
                
                # 3. Filter: Keep only if N content <= 5%
                if n_ratio <= MAX_N_RATIO:
                    # 4. Add spaces between characters (e.g., "A T G C ...")
                    spaced_seq = " ".join(list(subseq))
                    
                    # Write to file (one sequence per line)
                    f_out.write(spaced_seq + "\n")
                    count_saved += 1
                
                # 5. Calculate next step with random overlap
                # Next Start = Current End - Random Overlap
                overlap = random.randint(OVERLAP_MIN, OVERLAP_MAX)
                step = SEQ_LEN - overlap
                current_pos += step

    print(f"\nProcessing complete!")
    print(f"Total sequences saved: {count_saved}")
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Genome Segmentation Tool (Fixed length with random overlap and spacing)")
    
    parser.add_argument("-i", "--input", required=True, help="Path to input FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Path to output TXT file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    process_genome(args.input, args.output)

if __name__ == "__main__":
    main()