import sys
if __name__ == "__main__":
    dataset = sys.argv[1].lower()
    dataset_config = sys.argv[2].lower()
    print(len(sys.argv))
    if len(sys.argv) > 5:
        target = sys.argv[4]
        set_type = sys.argv[5]
        print(target)