import random
import argparse
from Bio import SeqIO

# Read the FASTA file and get sequences of specified chromosomes
def read_chromosomes(fasta_file, valid_ids=None):
    if valid_ids is None:
        raise ValueError("You Must Provide Valid Chromosome IDs")
    chromosomes = {
        record.id: record.seq
        for record in SeqIO.parse(fasta_file, "fasta")
        if record.id in valid_ids
    }
    return chromosomes

def read_valid_ids(region_file):
    valid_ids = []
    with open(region_file, 'r') as file:
        for line in file:
            if line.startswith("NC_090"):
                parts = line.strip().split('\t')
                chrom = parts[0]
                valid_ids.append(chrom)
    return valid_ids

def calculate_n_percentage(sequence, seq_length):

    n_count = sequence.count('N')
    return (n_count / seq_length)

# Perform a full genome slicing, extracting sequences of specified length
def full_genome_slice(chromosomes, seq_length):
    valid_sequences = []
    for chrom_id, chrom_seq in chromosomes.items():
        if len(chrom_seq) < seq_length:
            continue
        # Slide through the chromosome and extract sequences of the specified length
        start_pos = 0
        while start_pos + seq_length < len(chrom_seq):
            sequence = chrom_seq[start_pos:start_pos + seq_length].upper()
            
            # # Check if the sequence contains any invalid characters (i.e., anything other than ATCG)
            if calculate_n_percentage(sequence, seq_length) < 0.05:
                valid_sequences.append((chrom_id, start_pos, start_pos + seq_length, sequence.reverse_complement()))
            start_pos = start_pos + seq_length
            
            # overlap_offset = random.randint(0, 512)
            # start_pos = start_pos + seq_length - overlap_offset

        last_sequence = chrom_seq[len(chrom_seq)-seq_length:len(chrom_seq)].upper()          
        valid_sequences.append((chrom_id, len(chrom_seq)-seq_length, len(chrom_seq), last_sequence.reverse_complement()))

    return valid_sequences

# Save the extracted sequences in TSV format
def save_sequences_to_tsv(sequences, output_file):
    with open(output_file, "w") as f:
        # f.write("sequence\n")  # Write header
        for _, _, _, sequence in sequences:
            sequence = " ".join(list(sequence))
            f.write(f"{sequence}\n")

# Save the extracted sequences information in BED format
def save_sequences_to_bed(sequences, bed_file):
    with open(bed_file, "w") as f:
        for chrom_id, start, end, _ in sequences:
            f.write(f"{chrom_id}\t{start}\t{end}\n")

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice the genome into sequences of specified length, and filter out invalid sequences")
    parser.add_argument("-i", "--input_fasta", type=str, required=True, help="Path to the input genome FASTA file")
    parser.add_argument("-r", "--region_file", type=str, required=True, help="Path to the input Region file containing chromosome informations")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output TSV file containing sequences")
    parser.add_argument("-b", "--bed_file", type=str, required=True, help="Path to the output BED file")
    parser.add_argument("-l", "--sequence_length", type=int, required=True, help="Length of the sequences to be extracted")
    # parser.add_argument("-c", "--chromosomes", type=str, required=True, help="Comma-separated list of chromosome IDs to include")
    args = parser.parse_args()

    # Set parameters
    input_fasta = args.input_fasta
    region_file = args.region_file
    output_file = args.output_file
    bed_file = args.bed_file
    sequence_length = args.sequence_length

    # Run program
    valid_ids = read_valid_ids(region_file)[-2:]
    print(valid_ids)
    # valid_ids = [f"NC_0634{i}.1" for i in range(34,43)] + [f"NC_0634{i}.1" for i in range(44,52)]
    # valid_ids = ["NC_063443.1","NC_063452.1"]
    chromosomes = read_chromosomes(input_fasta, valid_ids)  # Read specific chromosomes
    valid_sequences = full_genome_slice(chromosomes, sequence_length)  # Perform full slicing
    save_sequences_to_tsv(valid_sequences, output_file)  # Save sequences to TSV
    save_sequences_to_bed(valid_sequences, bed_file)  # Save sequence positions to BED
    
    print(f"Filtered sequences have been saved to {output_file}")
    print(f"Sequence information has been saved to {bed_file}")
