from .backend import load_extension


def main():
    extension = load_extension(verbose=True)
    print(f"SYCL device: {extension.device_name()}")


if __name__ == "__main__":
    main()
