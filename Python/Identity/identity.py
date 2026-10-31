import cv2
import mediapipe as mp
import numpy as np
from Pose import JOINT_NAMES, JointState, PoseDetector, PoseTracker, RawPose, PlayerPose, JointState

MIN_CONF = 0.6
FALLBACK_TORSO_PIXELS = 150.0
POSE_MATCH_GATE = 0.9       # mean joint distance / torso length
DUPLICATE_GATE = 0.2        # same-person duplicate detection threshold
FACE_SCORE_MARGIN = 0.05    # tune against real template-score outputs
FALLBACK_SCALE = 150.0
FACE_WEIGHT = 1.0           # how much a face counts against position when matching to last frame's players
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

def poseDistance(pose1: RawPose, pose2: RawPose):
    usable = (pose1.confidence >= MIN_CONF) & (pose2.confidence >= MIN_CONF)
 
    if not usable.any():
        return float("inf")
 
    d = np.linalg.norm(pose1.points[usable] - pose2.points[usable], axis=1).mean()
 
    # Estimate the scale using the average skeleton size.
    scale = (poseScale(pose1) + poseScale(pose2)) / 2
 
    return float(d / max(scale, 1.0))


def poseScale(pose: RawPose):
    usable = pose.confidence >= MIN_CONF

    if usable.sum() < 2:
        return FALLBACK_SCALE

    points = pose.points[usable]

    width = np.ptp(points[:, 0])
    height = np.ptp(points[:, 1])

    return max(float(width), float(height), 1.0)


class Identity:

    def __init__(self, maxPlayers=2, faceScoreMargin=FACE_SCORE_MARGIN, imageSize=None):
        self.maxPlayers = int(maxPlayers)
        self.faceScoreMargin = faceScoreMargin
        self.imageSize = imageSize
        self.missingIdList = []
        self.known = {}

    def _head(self, pose):
        return pose.points[NOSE] if pose.confidence[NOSE] >= MIN_CONF else None

    def _face_center(self, face):

        if face.bbox is None or len(face.bbox) < 4:
            return None

        x, y, w, h = map(float, face.bbox[:4])
        if self.imageSize is not None:
            x, y, w, h = x * self.imageSize[0], y * self.imageSize[1], w * self.imageSize[0], h * self.imageSize[1]


        return np.array([
            x + w / 2,
            y + h / 2
        ])
    
    def _scores(self, face):
        """(score for template 1, score for template 2), or None if the face has none yet."""
        scores = getattr(face, "template_scores", None)
        return None if scores is None else tuple(scores)
 
    def _faceGap(self, face):
        """How clearly the two scores tell the players apart (0 = no idea)."""
        scores = self._scores(face) if face is not None else None
        return 0.0 if scores is None else abs(scores[0] - scores[1])
 
    def _faceVote(self, face):
        """1 or 2 if the scores clearly say which player this face is, else None."""
        if self._faceGap(face) < self.faceScoreMargin:
            return None
        scores = self._scores(face)
        return 1 if scores[0] > scores[1] else 2
 
    def _faceCost(self, face, pid):
        """Added to the matching cost: negative if the face supports player `pid`, positive if it says otherwise."""
        scores = self._scores(face) if face is not None else None
        if scores is None or pid not in (1, 2):
            return 0.0
        return FACE_WEIGHT * (scores[2 - pid] - scores[pid - 1])


    def _remove_duplicates(self, poses):
        duplicates = set()
        confidence = []

        for pose in poses:
            usable = pose.confidence >= MIN_CONF

            if usable.any():
                confidence.append(
                    float(pose.confidence[usable].mean())
                )
            else:
                confidence.append(0.0)

        for i in range(len(poses)):
            if i in duplicates:
                continue

            for j in range(i + 1, len(poses)):
                if j in duplicates:
                    continue

                d = poseDistance(poses[i], poses[j])

                if d < DUPLICATE_GATE:

                    # Keep the pose with the higher confidence.
                    if confidence[i] >= confidence[j]:
                        duplicates.add(j)
                    else:
                        duplicates.add(i)
                        break

        return duplicates

    def _matchFaceInitial(self, poses, face, duplicates, taken=None):
        face_coord = self._face_center(face)
        if face_coord is None:
            return None

        assigned = taken if taken is not None else set()        
        candidates = []

        for i, pose in enumerate(poses):
            if i in duplicates or i in assigned:
                continue

            head = self._head(pose)
            if head is None:
                continue

            d = np.linalg.norm(head - face_coord)

            if d <= FALLBACK_SCALE:
                candidates.append((d, i))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][1]

    def _bestInitialFaces(self, faces):
        scored = [(f, self._scores(f)) for f in faces if self._scores(f) is not None]
        best_pair = {}
        best_score = float("-inf")
 
        for i, (face1, scores1) in enumerate(scored):
            for j, (face2, scores2) in enumerate(scored):
                if i == j:
                    continue
 
                score = scores1[0] + scores2[1]
 
                if score > best_score:
                    best_score = score
                    best_pair[1] = face1
                    best_pair[2] = face2
 
        return best_pair

    def _getNextId(self, taken):
        for pid in range(1, self.maxPlayers + 1):

            if pid not in self.known and pid not in taken:

                return pid

        return None



    def _closestMissing(self, pose, taken):

        missing = self._missingIds(taken)
        if not missing:

            return None

        here = center(pose)

        pid = min(missing, key=lambda pid: np.linalg.norm(self.known[pid] - here))

        if np.linalg.norm(self.known[pid] - here) > 2 * FALLBACK_SCALE:
            return None

        return pid


    def _missingIds(self, taken):
        missing = []
        for k in self.known.keys():
            if k not in taken:
                missing.append(k)
        return missing

    def _matchFaces(self, poses, ids, faces, facesMatched=None):
        if facesMatched is None:
            facesMatched = {}
        for j,i in enumerate(ids):
            if i is not None and i not in facesMatched.keys():
                facesMatched[i] = self._matchFace(poses[j], faces)
        return facesMatched


    def _matchFace(self, pose, faces):
        bestMatch = None
        head_cord = self._head(pose)
        if head_cord is None:
            return None
        for f in faces:
            face_coord = self._face_center(f)
            if face_coord is None:
                continue
            dis = np.linalg.norm(head_cord-face_coord)
            if dis < FALLBACK_SCALE:
                if bestMatch is None or dis < bestMatch[1]:
                    bestMatch = (f, dis)
        if bestMatch is None:
            return None
        return bestMatch[0]

    def _poseIdentityExists(self, poses, previous, duplicates, taken, ids, faceOfPose=None):
        pairs = []

        for i, pose in enumerate(poses):
            if i in duplicates or ids[i] is not None:
                continue
            for player in previous:
                if player.id not in range(1,self.maxPlayers+1) or player.id in taken:
                    continue
                d = distance(pose, player)
                if d <= POSE_MATCH_GATE:
                    face = faceOfPose.get(i) if faceOfPose else None
                    pairs.append((d + self._faceCost(face, player.id),i,player.id))
        pairs.sort(key=lambda pair: pair[0])
    
        for cost, i, pid,  in pairs:
            if ids[i] is None and pid not in taken:
                ids[i] = pid
                taken.add(pid)
        
        return ids, taken

    def assign(self, poses, previousPoses, faces):
        ids = [None] * len(poses)
        duplicates = self._remove_duplicates(poses)
        taken = set()
        facesMatched = {}
        assigned = set()

        if not self.known:
            best_faces = self._bestInitialFaces(faces)

            if not best_faces:
                return ids, facesMatched
            hConfP1, hConfP2 = best_faces[1], best_faces[2]

            bestP1 = self._matchFaceInitial(poses, hConfP1, duplicates)
            if bestP1 is not None:
                ids[bestP1] = 1
                assigned.add(bestP1)
                taken.add(1)
                self.known[1] = center(poses[bestP1])

            bestP2 = self._matchFaceInitial(poses, hConfP2, duplicates, assigned)
            
            if bestP2 is not None:
                ids[bestP2] = 2
                taken.add(2)
                self.known[2] = center(poses[bestP2])
            
            return ids, (hConfP1, hConfP2)
        else:
            if self.missingIdList and faces:
                for m in self.missingIdList:
                    hConfF = None
                    for f in faces:
                        if self._faceVote(f) != m or any(f is g for g in facesMatched.values()):
                            continue
                        if hConfF is None or self._scores(f)[m - 1] > self._scores(hConfF)[m - 1]:
                            hConfF = f
                    if hConfF is None:
                        continue

                    bestP = self._matchFaceInitial(poses, hConfF, duplicates,assigned)
                    if bestP is not None:
                        ids[bestP] = m
                        taken.add(m)
                        self.known[m] = center(poses[bestP])
                        facesMatched[m] = hConfF
                        assigned.add(bestP)

            faceOfPose = {i: self._matchFace(pose, faces) for i, pose in enumerate(poses)}        
            ids, taken = self._poseIdentityExists(poses, previousPoses, duplicates, taken, ids)
            leftover = [i for i in range(len(poses)) if ids[i] is None and i not in duplicates]
            leftover.sort(key=lambda i: (-self._faceGap(faceOfPose[i]), center(poses[i])[0]))
            for i in leftover:
                vote = self._faceVote(faceOfPose[i])
                pid = vote if vote is not None and vote not in taken and vote <= self.maxPlayers else None
                if pid is None:
                    pid = self._closestMissing(poses[i], taken)
                if pid is None:
                    pid = self._getNextId(taken)
                if pid is None:
                    continue
                ids[i] = pid
                taken.add(pid)


            for pid, pose in zip(ids, poses):
                if pid is not None:
                    self.known[pid] = center(pose)
            self.missingIdList = self._missingIds(taken)


            facesMatched = self._matchFaces(poses, ids, faces, facesMatched)

            return ids, facesMatched

