import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from tracker import Tracker


def make_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640,360))
    if not writer.isOpened():
        raise RuntimeError("Could not create demo video with MJPG codec.")
    try:
        for i in range(140):
            frame = np.full((360,640,3), (28,23,18), dtype=np.uint8)
            x = int(45 + i*3.8)
            y = int(555 - i*3.6)
            cv2.rectangle(frame,(x,100),(x+36,140),(0,220,0),-1)
            cv2.rectangle(frame,(y,240),(y+36,280),(220,0,0),-1)
            cv2.putText(frame,"SYNTHETIC SHAPES DEMO | no learned detector",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def process_video(source, output, detector, stride=2, line_position=0.5, progress=None, max_frames=18000):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Cannot decode this video. Try a short MP4 or AVI file.")
    writer = None
    start = time.perf_counter()
    decoded, processed, peak = 0, 0, 0
    try:
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            raise ValueError("Video has invalid dimensions.")
        fps = fps if 0 < fps < 240 else 25.0
        scale = min(1, 960/width, 720/height)
        width, height = int(width*scale)//2*2, int(height*scale)//2*2
        if min(width,height) < 2:
            raise ValueError("Video dimensions are too small.")
        tracker = Tracker(width, line_position, max_distance=max(40,width*0.15))
        writer = cv2.VideoWriter(str(output/'annotated.avi'),cv2.VideoWriter_fourcc(*"MJPG"),fps/stride,(width,height))
        if not writer.isOpened(): raise RuntimeError("MJPG video writer is unavailable.")
        with (output/'tracks.csv').open('w',newline='',encoding='utf-8') as tf, (output/'crossings.csv').open('w',newline='',encoding='utf-8') as ef:
            tracks_csv = csv.DictWriter(tf,fieldnames=['frame','time_seconds','track_id','label','confidence','x1','y1','x2','y2'])
            events_csv = csv.DictWriter(ef,fieldnames=['frame','track_id','label','direction'])
            tracks_csv.writeheader(); events_csv.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok: break
                if decoded >= max_frames:
                    raise ValueError("Video exceeds the 18000-frame processing limit. Use a shorter clip.")
                frame_number = decoded; decoded += 1
                if frame_number % stride: continue
                frame = cv2.resize(frame,(width,height))
                visible, events = tracker.update(detector.detect(frame,frame_number),frame_number)
                processed += 1; peak = max(peak,len(visible))
                for event in events: events_csv.writerow(event)
                for t in visible:
                    values = dict(zip(['x1','y1','x2','y2'],[round(v,2) for v in t.box]))
                    tracks_csv.writerow(dict(frame=frame_number,time_seconds=round(frame_number/fps,3),track_id=t.id,label=t.label,confidence=round(t.confidence,4),**values))
                    color = ((t.id*53)%180+60,(t.id*97)%180+60,(t.id*139)%180+60)
                    x1,y1,x2,y2 = map(int,t.box)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    label = f"{t.label} #{t.id}"
                    label_width = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)[0][0]
                    label_x = max(0,min(x1,width-label_width-4))
                    cv2.putText(frame,label,(label_x,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                    if len(t.path)>1:
                        cv2.polylines(frame,[np.array(t.path,dtype=np.int32).reshape(-1,1,2)],False,color,2)
                line_x = int(width*line_position)
                cv2.line(frame,(line_x,0),(line_x,height),(80,190,250),2)
                counts = tracker.summary()
                cv2.putText(frame,f"Frame {frame_number} | Visible {len(visible)} | Tracks {sum(counts['unique_tracks'].values())}",(10,height-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(240,240,240),1)
                writer.write(frame)
                last_annotated = frame
                if processed % 5 == 0 or processed == 1:
                    cv2.imwrite(str(output/'preview.jpg'),frame)
                    if progress:
                        progress({"processed_frames": processed,"decoded_frames": decoded,"total_frames": total,"percent": min(99,round(decoded/total*100,1)) if total>0 else None,"counts": counts})
            if processed == 0: raise ValueError("Video contains no readable frames.")
            cv2.imwrite(str(output/'preview.jpg'),last_annotated)
        summary = {**tracker.summary(),"decoded_frames":decoded,"processed_frames":processed,"peak_visible_tracks":peak,"source_fps":fps,"stride":stride,"line_position":line_position,"processing_seconds":round(time.perf_counter()-start,3),"mode":type(detector).__name__,"note":"Track IDs can fragment or switch; counts are estimates, not ground-truth unique objects."}
        (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        return summary
    finally:
        cap.release()
        if writer is not None: writer.release()
