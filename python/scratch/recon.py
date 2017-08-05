import glob
from os import path
from mvshape import eval_out_reader
from mvshape import mve
from mvshape.proto import dataset_pb2

import blosc


def main():
    split_file = "/data/mvshape/out/splits/shrec12_examples_opo/test.bin"
    # basedir = "/data/mvshape/out/tf_out/depth_6_views/object_centered/tensors/0040_000011400/NOVELVIEW/"
    # tensors = eval_out_reader.read_tensors(basedir=basedir)
    #
    #
    with open(split_file, mode='rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)

    examples = dataset_pb2.Examples()
    examples.ParseFromString(decompressed)
    #
    #
    # print(tensors)
    #
    # print(examples)
    # print(examples.split_name)
    # print(dataset_pb2.Split.Name(examples.split_name))
    # print(examples.examples[0].multiview_depth.mesh_filename)
    # print(examples.examples[0].multiview_depth.eye[:])
    # print(examples.examples[0].multiview_depth.up[:])
    # print(examples.examples[0].multiview_depth.lookat[:])
    #
    # print(tensors.keys())

    # dirnames = sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/object_centered/recon/0040_000011400/*"))
    dirnames = sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/viewer_centered/recon/0040_000011685/*"))
    for dirname in dirnames:
        dirnames2 = sorted(glob.glob(path.join(dirname, '*')))
        for item in dirnames2[::3]:
            assert path.isdir(item)

            ply_files = glob.glob(path.join(item, "view_*.ply"))

            if path.isfile(path.join(item, "fssr_recon_clean.ply")):
                print("skipping", item)
            else:
                outfile = mve.fssr_pcl_files(ply_files, scale=0.3)
                outfile_clean = mve.meshclean(outfile, threshold=0.1)
                print(outfile_clean)



if __name__ == '__main__':
    main()
