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


def between_meshes(gt_file, recon_file):
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
    return dist


def between_mesh_and_pcl(gt_file, pcl_file):
    p, stdout, stderr = run_command([
        '/home/daeyun/git/mvshape/cmake-build-release/cpp/apps/mesh_distance',
        '--mesh1={}'.format(gt_file),
        '--pcl={}'.format(pcl_file),
    ])

    m = re.search(r'MEAN RMS PCL TO MESH: ([0-9\.e\-]+)', stderr)
    if m is None:
        print(stdout)
        print(stderr)
        raise RuntimeError()
    pcl_to_mesh = float(m.group(1))

    m = re.search(r'MEAN RMS MESH TO PCL: ([0-9\.e\-]+)', stderr)
    if m is None:
        print(stdout)
        print(stderr)
        raise RuntimeError()
    mesh_to_pcl = float(m.group(1))

    return pcl_to_mesh, mesh_to_pcl
