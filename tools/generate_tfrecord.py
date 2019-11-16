import argparse

from libs.datatool import ImageData


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='from which to generate TFRecord, folders or mxrec', default='mxrec')
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data', default='')
    parser.add_argument('--save_dir', type=str, help='path to save TFRecord file dir', default='')
    parser.add_argument('--thread_num', type=int, help='number of thread to progress data', default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cid = ImageData(img_size=args.image_size)
    if args.mode == 'folders':
        cid.write_tfrecord_from_folders(args.read_dir, args.save_dir)
    elif args.mode == 'mxrec':
        cid.write_tfrecord_from_mxrec(args.read_dir, args.save_dir, args.thread_num)
    else:
        raise('ERROR: wrong mode (only folders and mxrec are supported)')
