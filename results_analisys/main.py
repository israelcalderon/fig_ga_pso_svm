import argparse

from results_analisys.v1 import analisys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--means_file",
                      help="means dataset location",
                      default="/usr/src/app/db/means.csv")
    parser.add_argument("--std_file",
                      help="standar deviation dataset location",
                      default="/usr/src/app/db/std.csv")
    args = parser.parse_args()
    analisys.run(args.means_file, args.std_file)
