import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store', dest='test', default=1)
    parser.add_argument('--feature', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.feature:
        print(args.test)