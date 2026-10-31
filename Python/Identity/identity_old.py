import cv2
import mediapipe as mp
import numpy as np

from Pose import JOINT_NAMES, PoseDetector, PoseTracker, RawPose, PlayerPose, JointState
from Pose.motion import torso_length
from Pose.tracker import _mean_distance
from Face.faceTracking import FaceTracker

GATE = 0.6
SAME = 0.08
FALLBACK_SCALE = 150.0
MIN_CONF = 0.6
DUPLICATE_DISTANCE = 0.2 
NOSE = next(i for i, name in enumerate(JOINT_NAMES) if "nose" in str(name).lower())

def center(pose):
#     returns the center of the skeleton
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

def meanDistance(a, b):
    """Mean distance in pixels between the joints two poses both have confidently."""
    common = (a.confidence >= MIN_CONF) & (b.confidence >= MIN_CONF)
    if not common.any():
        return float("inf")
    return float(np.linalg.norm(a.points[common] - b.points[common], axis=1).mean())
class Identity:
    def __init__(self, maxPlayers = 2,):
         self.maxPlayers = maxPlayers
         self.nextId = 0
         self.known = {}

    def _getNextId(self) -> int:
         for pid in range(1, self.maxPlayers +1):
              if pid not in self.known.keys():
                   return pid
         return None


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

    def _missingId(self, pose, taken):
         missing = [pid for pid in self.known.keys() if pid not in taken]
         if not missing:
              return None
         here = center(pose)
         pid = (min(missing, key=lambda pid: np.linalg.norm(self.known[pid]-here)))

         if (np.linalg.norm(self.known[pid] - here)) > 2*FALLBACK_SCALE:
              return None
         return pid

    
    def head(self, pose):
        """Where the player's head (nose) is in pixels, or None if the nose is not confidently seen."""
        return pose.points[NOSE] if pose.confidence[NOSE] >= MIN_CONF else None

    def _faceIndex(self, pose, faces):
         "face of whose box holds this pose's head (the closest centere if several, or None"
         head = self.head(pose)
         if head is None:
              return None
         best, bestF = None, None
         for i, face in enumerate(faces):
              x, y, w, h = face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3]
              dx, dy = head[0] - (x + w / 2), head[1] - (y + h / 2)
              if abs(dx) <= 0.6 * w and abs(dy) <= 0.6 * h and (best is None or np.hypot(dx, dy) < best):
                   best, bestF = np.hypot(dx, dy), face
         return bestF

    def _deduplicate(self, poses):
          on = self.config.filter.confidence_on
          kept: list[RawPose] = []
          for pose in sorted(poses, key=lambda p: -p.confidence.mean()):
              scale = torso_length(pose.points, pose.confidence >= on) or FALLBACK_SCALE
              if all(_mean_distance(pose, other, on) / scale > self.config.duplicate_distance for other in kept):
                  kept.append(pose)
          return kept

    def _duplicates(self, poses):
         kept, duplicates = [], set()

         for i in sorted(range(len(poses)), key=lambda i: -poses[i].confidence.mean()):
              scale = torso_length(poses[i].points, poses[i].confidence >= MIN_CONF) or FALLBACK_SCALE
              if all(meanDistance(poses[i], poses[j]) / scale > DUPLICATE_DISTANCE for j in kept):
                   kept.append(i)
              else:
                   duplicates.add(i)
         return duplicates
    
    def assign(self, poses, previousPoses, faces) -> list:
     # Poses should be passed through deduplicate already before hand before being passed through here.
     duplicates = self._duplicates(poses)
     owners = [None if (j:= self._faceIndex(pose, faces)) is None else faces[j] for pose in poses]
     ids, taken = self._poseIdentityExists(poses, previousPoses, owners, duplicates)

     leftover = [i for i in range(len(poses)) if ids[i] is None and i not in duplicates]
     leftover.sort(key=lambda i: (faceMargin(owners[i]), center(poses[i])[0]))
     for i in leftover:
        vote = faceVote(owners[i])
        pid = vote if vote is not None and vote not in taken and vote <=self.maxPlayers else None
        if pid is None:
             pid = self._missingId(poses[i])



        
     #    if self.known.keys() is None:
     #         for i, pose in enumerate(keptPoses):
     #              if pose is not None:
     #                pid = self._getNextId
     #                ids[i] = pid
     #                self.known[pid] = self.heads(pose)
     #         faces = faceObj.initialLocations(self.known)
     #         idFaces = self.matchFaces(poses, ids, faces)
     #         return ids
     #    ids, taken = self._poseIdentityExists(keptPoses, previousPoses)

        for i, pose in enumerate(poses):
             if ids[i] is not None:
                  continue
             pid = self._missingId(pose, taken)
             if pid is None:
               pid = self._getNextId()
             ids[i] = pid
             taken.add(pid)       

        for pid, pose in zip(ids, poses):
             if pid is not None:
               self.known[pid] = center(pose)

        return ids

    def matchFaces(self, poses, ids, faces):
         """For each face, the id of the player whose head is inside its box, or None."""
         result = [None] * len(faces)
         for pose, pid in zip(poses, ids):
              if pid is not None and (j := self._faceIndex(pose, faces)) is not None:
                   result[j] = pid


    def heads(self, poses, ids):
         return{pid: (float(pose.points[NOSE][0]), float(pose.points[NOSE][1]))
                            for pid, pose in zip(ids, poses)
                            if pid is not None and pose.confidence[NOSE]>= MIN_CONF}

