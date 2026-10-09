import argparse
from pathlib import Path
from detectors import CLASSES, DemoDetector, YoloDetector
from video import make_demo, process_video

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Track objects in a local video.')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('runs/cli'))
    parser.add_argument('--classes',nargs='+',choices=list(CLASSES),default=['person','car'])
    parser.add_argument('--stride',type=int,choices=range(1,6),default=2)
    args=parser.parse_args()
    if not args.demo and args.source is None: parser.error('Provide --source or --demo')
    if args.demo:
        source=make_demo(args.output/'source.avi'); detector=DemoDetector()
    else:
        source=args.source; detector=YoloDetector(args.classes)
    print(process_video(source,args.output,detector,stride=args.stride))
