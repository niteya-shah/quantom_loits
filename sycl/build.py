from .backend import load_extension


def main():
    load_extension(verbose=True, allow_incomplete=True)


if __name__ == "__main__":
    main()
