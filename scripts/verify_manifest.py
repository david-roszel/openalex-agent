"""Verify the released file inventory without reading private configuration."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'PUBLIC_MANIFEST.json').read_text())
    for record in manifest['included_files']:
        relative=Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path=root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Redirected release file')
        data=path.read_bytes()
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Release checksum mismatch: '+relative.as_posix())
    return dict(verified=True,files=len(manifest['included_files']),manifest_self_checksum_excluded=True)


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1])))
