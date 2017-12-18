import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()

def main():
    print(args)
    print(args.resume)

if __name__ == "__main__":
    main()
