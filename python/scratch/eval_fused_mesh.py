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
    basedir = '/data/mvshape/out/recon/pytorch_mv6'
    results = collections.defaultdict(list)
    for tag in ['NOVELVIEW', 'NOVELMODEL', 'NOVELCLASS']:
        for i in range(0, 600, 7):
            example_dir = path.join(basedir, tag, '{:04d}'.format(i))
            gt_mesh = path.join(example_dir, 'gt.off')
            original = path.join(example_dir, 'recon/fssr_recon_clean.ply')
            fused = path.join(example_dir, 'recon/fssr_recon_clean.ply.fused.ply')

            p, stdout, stderr = run_command([
                '/home/daeyun/git/mvshape/cmake-build-release/cpp/apps/mesh_distance',
                '--mesh1={}'.format(gt_mesh),
                '--mesh2={}'.format(fused), ## change this
            ])
            # print(stderr)
            m = re.search(r'MEAN RMS: ([0-9\.]+)', stderr)
            if m is None:
                print(stdout)
                print(stderr)
                raise RuntimeError()
            dist = float(m.group(1))
            results[tag].append(dist)

    print('#############################')
    for tag in ['NOVELVIEW', 'NOVELMODEL', 'NOVELCLASS']:
        result_string = '{}:  mean {:.5f}, median {:.5f}, std {:.5f}'.format(tag, np.array(results[tag]).mean(), np.median(results[tag]), np.array(results[tag]).std())
        print(result_string)


if __name__ == '__main__':
    main()
