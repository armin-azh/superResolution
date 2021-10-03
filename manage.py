from argparse import (Namespace,
                      ArgumentParser)


def main(arguments: Namespace):
    pass


if __name__ == '__main__':
    parser = ArgumentParser()
    # start options
    # end options
    args = parser.parse_args()
    main(arguments=args)
