import math
from pathlib import Path
import threading
import uuid
from flask import Flask, jsonify, render_template, request, send_from_directory
from detectors import CLASSES, DemoDetector, YoloDetector
from video import make_demo, process_video

ROOT = Path(__file__).resolve().parent
ALLOWED = {".mp4",".avi",".mov",".mkv"}
ARTIFACTS = {"preview.jpg","annotated.avi","summary.json","tracks.csv","crossings.csv"}


def create_app(run_root=None, detector_factory=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 100*1024*1024
    root = Path(run_root or ROOT/'runs'); root.mkdir(parents=True,exist_ok=True)
    jobs, lock = {}, threading.Lock()

    def execute(job_id, source, mode, labels, confidence, stride, line):
        def update(values):
            with lock: jobs[job_id].update(values)
        try:
            update({"status":"loading","message":"Loading detector. First YOLO run may download model weights."})
            detector = detector_factory(mode,labels,confidence) if detector_factory else (DemoDetector() if mode=='demo' else YoloDetector(labels,confidence))
            update({"status":"processing","message":"Analyzing video"})
            result = process_video(source,root/job_id,detector,stride,line,progress=update)
            update({"status":"done","percent":100,"message":"Complete","summary":result})
        except Exception as error:
            update({"status":"error","message":str(error)})

    @app.get('/')
    def index(): return render_template('index.html',classes=list(CLASSES))

    @app.get('/health')
    def health(): return jsonify(status='ok',detector='loaded on demand')

    @app.post('/api/jobs')
    def create_job():
        mode = request.form.get('mode','yolo')
        labels = request.form.getlist('classes')
        try:
            if mode not in {'yolo','demo'}: raise ValueError('Invalid processing mode.')
            stride = int(request.form.get('stride','2'))
            confidence = float(request.form.get('confidence','0.35'))
            line = float(request.form.get('line','0.5'))
            if stride not in {1,2,3,4,5}: raise ValueError('Stride must be 1 through 5.')
            if not math.isfinite(confidence) or not 0.1<=confidence<=0.9: raise ValueError('Confidence must be 0.1 through 0.9.')
            if not math.isfinite(line) or not 0.1<=line<=0.9: raise ValueError('Line position must be 0.1 through 0.9.')
            if mode=='yolo' and (not labels or not set(labels).issubset(CLASSES)): raise ValueError('Select at least one supported class.')
            upload = request.files.get('video')
            if mode=='yolo' and (not upload or Path(upload.filename or '').suffix.lower() not in ALLOWED):
                raise ValueError('Upload an MP4, AVI, MOV, or MKV video.')
        except (ValueError,TypeError) as error:
            return jsonify(error=str(error)),400
        with lock:
            if any(j['status'] in {'queued','loading','processing'} for j in jobs.values()):
                return jsonify(error='A video is already processing. Wait for it to finish.'),409
            if len(jobs)>=20:
                return jsonify(error='Session limit reached. Restart the app to process more clips.'),429
            job_id = uuid.uuid4().hex
            jobs[job_id] = {"id":job_id,"status":"queued","percent":0,"message":"Preparing video","mode":mode}
        folder=root/job_id; folder.mkdir()
        try:
            if mode=='demo': source=make_demo(folder/'source.avi')
            else:
                source=folder/('source'+Path(upload.filename).suffix.lower())
                upload.save(source)
        except Exception as error:
            with lock: jobs[job_id].update(status='error',message=str(error))
            return jsonify(error='Could not save the input video.'),500
        threading.Thread(target=execute,args=(job_id,source,mode,labels,confidence,stride,line),daemon=True).start()
        return jsonify(id=job_id),202

    @app.get('/api/jobs/<job_id>')
    def status(job_id):
        with lock:
            if job_id not in jobs: return jsonify(error='Job not found'),404
            return jsonify(dict(jobs[job_id]))

    @app.get('/api/jobs/<job_id>/files/<name>')
    def artifact(job_id,name):
        with lock:
            if job_id not in jobs or name not in ARTIFACTS: return jsonify(error='File not found'),404
            if name!='preview.jpg' and jobs[job_id]['status']!='done': return jsonify(error='Result is not ready'),409
        return send_from_directory(root/job_id,name,as_attachment=name!='preview.jpg')

    @app.errorhandler(413)
    def too_large(error): return jsonify(error='Video exceeds the 100 MB upload limit.'),413

    return app

if __name__=='__main__':
    create_app().run(host='127.0.0.1',port=5004,debug=False)
