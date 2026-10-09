import unittest
from tracker import Tracker, iou

def detection(x,label='car'):
    return {"box":[x,30,x+20,50],"label":label,"confidence":0.9}

class TrackingTests(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[20,20,30,30]),0)

    def test_confirmation_and_stable_id(self):
        t=Tracker(200,min_hits=3)
        for i,x in enumerate([10,15,20]): visible,_=t.update([detection(x)],i)
        self.assertEqual(visible[0].id,1)
        self.assertEqual(t.summary()['unique_tracks'],{'car':1})

    def test_occlusion_recovery(self):
        t=Tracker(200,min_hits=1)
        t.update([detection(10)],0);t.update([detection(15)],1)
        t.update([],2)
        visible,_=t.update([detection(25)],3)
        self.assertEqual(visible[0].id,1)

    def test_class_gating(self):
        t=Tracker(200,min_hits=1)
        t.update([detection(10)],0)
        visible,_=t.update([detection(10,'person')],1)
        self.assertEqual(visible[0].id,2)

    def test_expiration(self):
        t=Tracker(200,max_missed=1,min_hits=1)
        t.update([detection(10)],0);t.update([],1);t.update([],2)
        visible,_=t.update([detection(10)],3)
        self.assertEqual(visible[0].id,2)

    def test_crossings_and_hysteresis(self):
        t=Tracker(200,min_hits=1)
        all_events=[]
        for i,x in enumerate([50,70,89,90,91,110,90,70,110]):
            _,events=t.update([detection(x)],i);all_events+=events
        self.assertEqual(len(all_events),2)
        self.assertEqual({e['direction'] for e in all_events},{'left_to_right','right_to_left'})

    def test_two_tracks_do_not_share_detection(self):
        t=Tracker(200,min_hits=1)
        t.update([detection(10),detection(70)],0)
        visible,_=t.update([detection(15)],1)
        self.assertEqual(len(visible),1)
        self.assertEqual(visible[0].id,1)

    def test_distance_gate(self):
        t=Tracker(500,min_hits=1,max_distance=30)
        t.update([detection(10)],0)
        visible,_=t.update([detection(200)],1)
        self.assertEqual(visible[0].id,2)

if __name__=='__main__': unittest.main()
