import argparse
import sys

parser = argparse.ArgumentParser(description='save results.')

parser.add_argument('--a', type=float, default=0,
                        help='trade-off parameters of distillation.')
args = parser.parse_args()

if args.a == 1:
    print(args.a)
    sys.exit()
elif args.a == 2:
    print(args.a)

print('not exist.')