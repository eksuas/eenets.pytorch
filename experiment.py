"""
Edanur Demir
07 Nov 2019
"""
import argparse
import csv

def init():
    parser = argparse.ArgumentParser(description='EENets experiments')
    parser.add_argument('--filename', type=str, help='file to be analyzed')
    return parser.parse_args()

def read(args):
    data = []
    keys = []
    with open(args.filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line, row in enumerate(csv_reader):
            if line == 0:
                keys = row
            else:
                data.append({col:row[id] for id, col in enumerate(keys)})
    return data

def analyze(data):
    # Some samples can be classified differently from resnet. How much are they better?
    compare_resnet = {"True":0, "False":0}
    increasing_conf = 0
    early_confident = 0
    late_confident = 0
    for sample in data:
        # Samples classified differently from the last exit
        if sample["actual_pred"] != sample["start_pred_seq"]:
            if sample["actual_pred"] == sample["target"]:
                compare_resnet["True"] += 1
                early_confident += 1
            elif sample["start_pred_seq"] == sample["target"]:
                compare_resnet["False"] += 1
            increasing_conf += sample["start_conf_seq"] < sample["actual_conf"]

        if sample["actual_exit"] > sample["start_exit_seq"]:
            if sample["actual_pred"] == sample["target"]:
                late_confident += 1
            increasing_conf += sample["start_conf_seq"] > sample["actual_conf"]

    print("Samples classified differently from the last exit")
    print(compare_resnet)
    print()
    print("The rate of early true confident samples: {:.2f}".format(early_confident/10000.))
    print("The rate of late true confident samples: {:.2f}".format(late_confident/10000.))
    print()
    print("The rate of samples whose confidence is increasing: {:.2f}".format(increasing_conf/10000.))


def main():
    """Main function of the program.

    The function loads the dataset and calls training and validation functions.
    """
    args = init()
    data = read(args)
    analyze(data)

if __name__ == '__main__':
    main()
