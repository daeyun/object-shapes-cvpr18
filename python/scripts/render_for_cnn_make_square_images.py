from mvshape import render_for_cnn_utils


def main():
    render_for_cnn_utils.truncate_images('/home/daeyun/git/RenderForCNN/data/syn_images_cropped/',
                                         '/data/render_for_cnn/data/syn_images_cropped_square/',
                                         ignore_overwrite=True)


if __name__ == '__main__':
    main()
