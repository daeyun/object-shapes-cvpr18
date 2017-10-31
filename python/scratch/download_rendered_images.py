import os
import numpy as np
from pprint import pprint
import re
import paramiko
import sys
from os import path


def count_files(ssh_client: paramiko.SSHClient, dirname, pattern, recursive=True):
    if not recursive:
        raise NotImplementedError()

    ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command(
            'find {} -mindepth 1 -type f -name "{}" -printf x | wc -c'.format(dirname, pattern))
    output = ssh_stdout.read().decode()
    m = re.search(r'\d+', output)
    count = int(m.group(0))
    return count



key = paramiko.ECDSAKey.from_private_key_file(path.expanduser('~/.ssh/keys/id_ecdsa.ics-titan'))
key1 = paramiko.ECDSAKey.from_private_key_file(path.expanduser('~/.ssh/keys/id_ecdsa.daeyuns-uci-nop'))

syn_images_dir = '~/shared/home/git/RenderForCNN/data/syn_images'
syn_images_dir1 = '~/git/RenderForCNN/data/syn_images'


hosts = [
    ('dshin0.ics.uci.edu', 'daeyun', key1, syn_images_dir1),
    ('rhea.ics.uci.edu', 'daeyuns', key, syn_images_dir),
    ('oceanus.ics.uci.edu', 'daeyuns', key, syn_images_dir),
    ('tethys.ics.uci.edu', 'daeyuns', key, syn_images_dir),
    ('themis.ics.uci.edu', 'daeyuns', key, syn_images_dir),
    ('cronus.ics.uci.edu', 'daeyuns', key, syn_images_dir),
]

clients = {}

completed = {}

for host, user, k, syn_images_dir in hosts:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, username=user, pkey=k)
    clients[host] = ssh
    print(host)
    print('-------------------------------')

    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('ls {}'.format(syn_images_dir))
    output = ssh_stdout.read().decode()
    synsets = re.findall(r'\d+', output)

    for synset in synsets:
        dirname = '{}/{}'.format(syn_images_dir, synset)
        count = count_files(ssh, dirname=dirname, pattern='*.png')
        print(synset, count)
        if count > 190000:
            if synset not in completed:
                completed[synset] = (dirname, count, host)
            else:
                if completed[synset][1] != count:
                    print('Mismatch: ', completed[synset], count)
    print()



pprint(completed)

assert len(completed) == 12, len(completed)


for synset in completed:
    dirname, count, host = completed[synset]
    remote_source = '{}:{}'.format(host, dirname)
    remote_source = remote_source.replace('dshin0.ics.uci.edu', 'dshin0')
    remote_source = remote_source.replace('ics.uci.edu', 'ics')
    local_target = '~/git/RenderForCNN/data/syn_images/'
    assert local_target.endswith('/')
    assert not remote_source.endswith('/')
    rsync_command = 'rsync -atvur --info=progress2 {} {}'.format(remote_source, local_target)
    print(rsync_command)

print(np.sum([v[1] for v in completed.values()]))




for host, _, _, _ in hosts:
    clients[host].close()
