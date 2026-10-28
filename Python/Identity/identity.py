import cv2
import mediapipe as mp
import numpy as np

from Pose import PoseDetector, PoseTracker, RawPose, PlayerPose, JointState


GATE = 0.6
SAME = 0.08
FALLBACK_SCALE = 150.0
MIN_CONF = 0.6

def center(pose):
#     returns the center of teh skeleton
    confident = pose.confidence >= MIN_CONF

    if confident.any():
        return pose.points[confident].mean(axis=0)

    return pose.points.mean(axis=0)


def distance(pose: RawPose, player: PlayerPose):
     usable = (player.state >= JointState.PREDICTED) & (pose.confidence >= MIN_CONF)
     if not usable.any():
          return float("inf")
     d = np.linalg.norm(pose.points[usable]-player.position[usable], axis=1).mean()
     return float(d/max(player.motion.scale or FALLBACK_SCALE, 1.0))

class Identity:
    def __init__(self, maxPlayers = 2, t = 0):
         self.maxPlayers = maxPlayers
         self.nextId =1
         self.known = {}

    def _getNextId(self) -> int:
         id = self.nextId
         self.nextId += 1
         return id


    def _poseIdentityExists(self, poses, previous):
          pairs = []
          for i, pose in enumerate(poses):
               for player in previous:
                    d = distance(pose, player)
                    if d <= GATE:
                         pairs.append((d,i,player.id))
          pairs.sort(key=lambda pair: pair[0])

          ids = [None] * len(poses)
          taken = set()
          for d, i, pid,  in pairs:
               if ids[i] is None and pid not in taken:
                    ids[i] = pid
                    taken.add(pid)
               
          return ids, taken

    
#     def _missingIds(self, taken):
#           missing = [pid for pid in self.known if pid not in taken]
#           if not missing:
#                return None
#           return missing

    def _missingId(self, pose, taken):
         missing = [pid for pid in self.known if pid not in taken]
         if not missing:
              return None
         here = center(pose)
         pid = (min(missing, key=lambda pid: np.linalg.norm(self.known[pid]-here)))

         if (np.linalg.norm(self.known[pid] - here)) > 2*FALLBACK_SCALE:
              return None
         return pid

    def assign(self, poses, previousPoses) -> list:
     # Poses should be passed through deduplicate already before hand before being passed through here.
        ids, taken = self._poseIdentityExists(poses, previousPoses)
     #    missing = self._missingIds(taken)

     
        for i, pose in enumerate(poses):
             if ids[i] is not None:
                  continue
             pid = self._missingId(pose, taken)
             if pid is None:
               pid = self._getNextId()
             ids[i] = pid
             taken.add(pid)

          #    if missing is not None:
          #         pid = self._missingId(pose, taken)
          #         ids[i] = pid
          #         taken.add(pid)
          #         missing.remove(pid)
          #         continue
             
             

        for pid, pose in zip(ids, poses):
             if pid is not None:
               self.known[pid] = center(pose)

        return ids