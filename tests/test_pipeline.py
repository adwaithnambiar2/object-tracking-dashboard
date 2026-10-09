import csv
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from app import create_app
from detectors import DemoDetector
from video import make_demo, process_video

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_synthetic_video_and_exports(self):
        source=make_demo(self.root/'demo.avi')
        result=process_video(source,self.root/'out',DemoDetector())
        self.assertEqual(result['unique_tracks'],{'demo_green':1,'demo_blue':1})
        self.assertEqual(sum(c['count'] for c in result['crossings']),2)
        self.assertEqual(result['processed_frames'],70)
        with (self.root/'out/tracks.csv').open() as f: self.assertGreater(len(list(csv.DictReader(f))),100)
        self.assertTrue((self.root/'out/annotated.avi').stat().st_size>1000)
        self.assertEqual(json.loads((self.root/'out/summary.json').read_text())['mode'],'DemoDetector')

    def test_invalid_video(self):
        bad=self.root/'bad.mp4';bad.write_bytes(b'not a movie')
        with self.assertRaises(ValueError): process_video(bad,self.root/'out',DemoDetector())

    def test_api_input_validation(self):
        c=create_app(self.root).test_client()
        self.assertEqual(c.get('/').status_code,200)
        self.assertEqual(c.get('/health').status_code,200)
        for data in [{},{'mode':'other'},{'mode':'demo','stride':'0'},{'mode':'demo','confidence':'nan'},{'mode':'demo','line':'1.5'}]:
            self.assertEqual(c.post('/api/jobs',data=data).status_code,400)
        self.assertEqual(c.get('/api/jobs/not-found').status_code,404)

    def test_async_demo_and_download(self):
        c=create_app(self.root).test_client()
        response=c.post('/api/jobs',data={'mode':'demo'})
        self.assertEqual(response.status_code,202)
        job=response.json['id']
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            state=c.get('/api/jobs/'+job).json
            if state['status'] in {'done','error'}: break
            time.sleep(0.05)
        self.assertEqual(state['status'],'done',state)
        with c.get(f'/api/jobs/{job}/files/summary.json') as download:
            self.assertEqual(download.status_code,200)
        self.assertEqual(c.get(f'/api/jobs/{job}/files/source.avi').status_code,404)

if __name__=='__main__': unittest.main()
