"""Local fixture tests; no GitHub calls or real experiment uploads."""
import json
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from backup_release import prepare, verify, check_assets, sha256, upload


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='auv_backup_test_')
        self.root = Path(self.temp.name)
        self.archive = self.root/'twophase_full_fixture.tar.gz'
        self.archive.write_bytes(bytes(range(100)))

    def tearDown(self): self.temp.cleanup()

    def download(self, files):
        folder = self.root/'download'; folder.mkdir()
        for p in files: (folder/p.name).write_bytes(p.read_bytes())
        return folder

    def test_single_file(self):
        manifest, files = prepare(self.archive, asset_limit=200)
        self.assertFalse(manifest['split'])
        folder = self.download(files)
        self.assertEqual(verify(folder).read_bytes(), self.archive.read_bytes())
        verify(folder)

    def test_split_and_reassemble(self):
        manifest, files = prepare(self.archive, asset_limit=60, chunk_size=30)
        self.assertTrue(manifest['split']); self.assertEqual(len(manifest['assets']),4)
        folder = self.download(files)
        self.assertEqual(verify(folder).read_bytes(), self.archive.read_bytes())
        verify(folder)

    def test_corrupt_part_rejected(self):
        manifest, files = prepare(self.archive, asset_limit=60, chunk_size=30)
        folder = self.download(files)
        (folder/manifest['assets'][0]['name']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError): verify(folder)

    def test_remote_metadata_and_digest(self):
        _, files = prepare(self.archive, asset_limit=200)
        assets = [dict(name=p.name,state='uploaded',size=p.stat().st_size,digest='sha256:'+sha256(p)) for p in files]
        self.assertEqual(check_assets(dict(assets=assets),files),[])
        assets[0]['size']-=1
        with self.assertRaises(RuntimeError): check_assets(dict(assets=assets),files)

    def test_path_traversal_rejected(self):
        _, files = prepare(self.archive, asset_limit=200)
        folder = self.download(files)
        path = folder/'backup_manifest.json'; value=json.loads(path.read_text())
        value['assets'][0]['name']='../outside.tar.gz'; path.write_text(json.dumps(value))
        with self.assertRaises(ValueError): verify(folder)

    def test_failed_upload_stays_draft_and_retry_skips_complete_asset(self):
        state={'release':None,'failed':False,'uploads':[]}
        def api(endpoint,method='GET',payload=None):
            if endpoint.endswith('auv_traj_exp_Two_Phase'):
                return {'permissions':{'push':True}}
            if method=='POST':
                state['release']=dict(payload,id=123,assets=[],html_url='https://example.invalid/backup')
                return copy.deepcopy(state['release'])
            if method=='PATCH':
                state['release'].update(payload)
                return copy.deepcopy(state['release'])
            if state['release'] is None:
                raise subprocess.CalledProcessError(1,['gh','api'],stderr='HTTP 404')
            return copy.deepcopy(state['release'])
        def run(command,**kwargs):
            path=Path(command[4])
            if path.name=='backup_manifest.json' and not state['failed']:
                state['failed']=True
                raise subprocess.CalledProcessError(1,command,stderr='synthetic transfer interruption')
            state['uploads'].append(path.name)
            state['release']['assets'].append(dict(name=path.name,state='uploaded',size=path.stat().st_size,digest='sha256:'+sha256(path)))
            return subprocess.CompletedProcess(command,0)
        with patch('backup_release.shutil.which',return_value='gh'), patch('backup_release.gh_api',side_effect=api), patch('backup_release.subprocess.run',side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError): upload(self.archive,'backup-fixture')
            self.assertTrue(state['release']['draft'])
            upload(self.archive,'backup-fixture')
        self.assertFalse(state['release']['draft'])
        self.assertEqual(state['uploads'].count(self.archive.name),1)


if __name__=='__main__': unittest.main(verbosity=2)
