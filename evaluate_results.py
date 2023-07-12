import argparse
import csv
import jiwer
import os
import sys

def evaluate_results(ground_truth_file, evaluated_file, output_folder):

    output_file = os.path.join(output_folder, os.path.basename(evaluated_file)[:-4] + "_eval.csv")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
        
    # Read the ground truth file into a dictionary
    ground_truth = {}
    with open(ground_truth_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            ground_truth[row[0]] = row[1:]

    # Read the text extraction file and calculate the WER for each line
    results = []
    with open(evaluated_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        header.insert(1, "sum")
        for row in reader:
            error_rates = []
            filename = row[0]
            texts = row[1:]
            ground_truth_texts = ground_truth.get(filename, [])
            for col in range(len(texts)):
                ground_truth_text, text = ground_truth_texts[col], texts[col]
                if ground_truth_text == "": ground_truth_text = "-"
                if text == "": text = "-"
                if col in (0, 1, 2, 3, 5, 6): # temporarily validading only some columns
                    error_rate = jiwer.wer(ground_truth_text, text)
                else:
                    error_rate = 0
                error_rates.append(error_rate)
            error_rates.insert(0, sum(error_rates))
            error_rates.insert(0, filename)
            results.append(error_rates)

    # Write the results to the output file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process PDF files and save results to a CSV file')
    parser.add_argument('ground_truth_file', help='path to the ground truth file')
    parser.add_argument('evaluated_file', help='path to the evaluation file')
    parser.add_argument('output_folder', help='path to the output file')    
    args = parser.parse_args()
    evaluate_results(args.ground_truth_file, args.evaluated_file, args.output_folder)

