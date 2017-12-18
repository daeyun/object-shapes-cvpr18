import glob
import numpy as np
import re
from os import path
import collections
from mvshape import eval_out_reader
from mvshape import mve
from mvshape.proto import dataset_pb2
from dshin import log
import subprocess
import time
import textwrap

import blosc


def run_command(command):
    log.debug(subprocess.list2cmdline(command))

    start_time = time.time()
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = p.wait()
    log.debug('return code: {}'.format(return_code))
    elapsed = time.time() - start_time

    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    if p.returncode != 0:
        exec_summary = textwrap.dedent(textwrap.dedent("""
        Command: {}
        Return code: {}

        stdout:
        {}

        stderr:
        {}
        """).format(
            subprocess.list2cmdline(command),
            p.returncode,
            stdout,
            stderr
        ))
        raise RuntimeError(exec_summary)

    log.info('{0} took {1:.3f} seconds.'.format(path.basename(command[0]), elapsed))

    return p, stdout, stderr


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


    # dirnames = sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/object_centered/recon/0040_000011400/*"))
    # # dirnames = sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/viewer_centered/recon/0040_000011685/*"))

    if True:
        all_dirnames = [
            sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/object_centered/recon/0040_000011400/*")),
            sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/viewer_centered/recon/0040_000011685/*"))
        ]
        for dirnames in all_dirnames:
            for dirname in dirnames:
                dirnames2 = sorted(glob.glob(path.join(dirname, '*')))
                for item in dirnames2:
                    assert path.isdir(item)

                    ply_files = glob.glob(path.join(item, "view_*.ply"))

                    if path.isfile(path.join(item, "fssr_recon_clean.ply")):
                        print("skipping", item)
                    else:
                        outfile = mve.fssr_pcl_files(ply_files, scale=0.25)
                        outfile_clean = mve.meshclean(outfile, threshold=0.9)
                        print(outfile_clean)

    if True:
        all_dirnames = [
            sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/object_centered/recon/0040_000011400/*")),
            sorted(glob.glob("/data/mvshape/out/tf_out/depth_6_views/viewer_centered/recon/0040_000011685/*"))
        ]
        for dirnames in all_dirnames:
            result = collections.defaultdict(list)
            for dirname in dirnames:
                print('#####################', dirname)
                experiment_tag = path.basename(dirname)
                print(experiment_tag)
                dirnames2 = sorted(glob.glob(path.join(dirname, '*')))
                for item in dirnames2:
                    assert path.isdir(item)
                    recon_file = path.join(item, "fssr_recon_clean.ply")

                    # TODO
                    if not path.isfile(recon_file):
                        print(recon_file, 'does not exist. skipping.')
                        continue

                    gt_file = path.join(item, "gt.off")
                    # print(gt_file)
                    assert path.isfile(recon_file)
                    assert path.isfile(gt_file)

                    try:
                        p, stdout, stderr = run_command([
                            '/home/daeyun/git/mvshape/cmake-build-release/cpp/apps/mesh_distance',
                            '--mesh1={}'.format(gt_file),
                            '--mesh2={}'.format(recon_file),
                        ])
                        # print(stderr)
                        m = re.search(r'MEAN RMS: ([0-9\.e\-]+)', stderr)
                        if m is None:
                            print(stdout)
                            print(stderr)
                            raise RuntimeError()
                        dist = float(m.group(1))
                        result[experiment_tag].append(dist)
                    except Exception as ex:
                        print(ex)

            print('##########################################')
            print('##########################################')
            print('##########################################')
            print('# RESULTS')
            for k, v in result.items():
                print(k, np.array(v).mean())
            print('# RESULTS (median)')
            for k, v in result.items():
                print(k, np.median(v))


if __name__ == '__main__':
    main()
