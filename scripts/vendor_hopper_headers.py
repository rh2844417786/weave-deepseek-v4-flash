#!/usr/bin/env python3
"""Development-only: materialize a pinned CUTLASS header closure from a local archive.

No network or package installation is performed. Server jobs consume the
already committed headers and verify manifest.json through hopper_candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile

COMMIT='f3fde58372d33e9a5650ba7b80fc48b3b49d40c8'
ARCHIVE_SHA256='2e5890306557bd87136e1d6914e4ee42536050b9116433848175520fd15991f2'
ROOTS=('cutlass/arch/barrier.h','cute/arch/copy_sm90_tma.hpp',
       'cute/arch/mma_sm90_gmma.hpp','cute/arch/mma_sm90_desc.hpp','cute/swizzle.hpp')
INCLUDES=re.compile(r'^\s*#\s*include\s*(["<])([^">]+)[">]',re.MULTILINE)


def materialize(archive,destination):
    archive,destination=Path(archive),Path(destination)
    if destination.exists():raise FileExistsError('fresh vendor destination required')
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_sha!=ARCHIVE_SHA256:raise ValueError('pinned CUTLASS archive SHA256 mismatch')
    prefix='cutlass-'+COMMIT+'/'
    with tarfile.open(archive,'r:gz') as tar:
        members={m.name[len(prefix):]:m for m in tar.getmembers() if m.isfile() and m.name.startswith(prefix)}
        def read(name):
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('unsafe archive path')
            return tar.extractfile(members[name]).read()
        queued=list(ROOTS);files={}
        while queued:
            name=queued.pop()
            if name in files:continue
            data=read('include/'+name);files[name]=data
            for delimiter,inc in INCLUDES.findall(data.decode()):
                candidates=([str(PurePosixPath(name).parent/inc)] if delimiter=='"' else [])+[inc]
                found=next((p for p in candidates if 'include/'+p in members),None)
                if found:queued.append(found)
                elif inc.startswith(('cute/','cutlass/')):raise ValueError('missing pinned include '+inc)
        license_text=read('LICENSE.txt')
    manifest={'schema_version':1,'upstream':{'repository':'https://github.com/NVIDIA/cutlass',
        'commit':COMMIT,'archive_sha256':archive_sha},
        'roots':list(ROOTS),'files':{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())}}
    for name,data in files.items():
        path=destination/'include'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (destination/'LICENSE.txt').write_bytes(license_text)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();value=materialize(args.archive,args.destination)
    print(json.dumps({'headers':len(value['files']),'upstream':value['upstream']}))
