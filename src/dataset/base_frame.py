from typing import Tuple, List, Dict, Any

import json
from PIL import Image
import numpy as np


class Frame:
    def __init__(self, *, color_path, depth_path=None, pose_path, K=None, **kwargs):
        self.color_path = color_path
        self.depth_path = depth_path
        self.pose_path = pose_path
        self.K = K

        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.color_path is not None:
            with Image.open(self.color_path) as img:
                self.color = img.copy()

        if self.depth_path is not None:
            with Image.open(self.depth_path) as img:
                self.depth = img.copy()
            if self.depth.size != self.color.size:
                self.depth = self.depth.resize(self.color.size, Image.NEAREST)
        else :
            self.depth = None

        if self.pose_path is not None:
            self.pose = np.loadtxt(self.pose_path)

        if self.K is not None:
            self.K_inv = np.linalg.inv(self.K)
        else:
            self.K_inv = None


class FramePair:
    def __init__(self, src_frame: Frame, tgt_frame: Frame):
        self.src_frame = src_frame
        self.tgt_frame = tgt_frame

        self.rpv = None
        self.tau_and_cpd = None

    def save(self, output_dir: str, level: str, sample_id: int):
        """
        Save the frame pair data to the specified output directory.
        """
        # Implement saving logic as needed, e.g., saving file paths, RPV, etc.
        actual_output_dir = output_dir / level / f"{sample_id:06d}"
        actual_output_dir.mkdir(parents=True, exist_ok=True)

        src_dir = actual_output_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir = actual_output_dir / "tgt"
        tgt_dir.mkdir(parents=True, exist_ok=True)

        # Save color images
        src_color_save_path = src_dir / self.src_frame.color_path.name
        self.src_frame.color.save(src_color_save_path)
        if self.src_frame.depth is not None:
            src_depth_save_path = src_dir / self.src_frame.depth_path.name
            self.src_frame.depth.save(src_depth_save_path)
        src_pose_save_path = src_dir / self.src_frame.pose_path.name
        np.savetxt(src_pose_save_path, self.src_frame.pose)

        tgt_color_save_path = tgt_dir / self.tgt_frame.color_path.name
        self.tgt_frame.color.save(tgt_color_save_path)
        if self.tgt_frame.depth is not None:
            tgt_depth_save_path = tgt_dir / self.tgt_frame.depth_path.name
            self.tgt_frame.depth.save(tgt_depth_save_path)
        tgt_pose_save_path = tgt_dir / self.tgt_frame.pose_path.name
        np.savetxt(tgt_pose_save_path, self.tgt_frame.pose)

        # Save intrinsic matrices if available
        if self.src_frame.K is not None:
            K_save_path = actual_output_dir / "K.txt"
            np.savetxt(K_save_path, self.src_frame.K)

        # Save RPV and other computed data if needed
        if self.rpv is not None:
            self.rpv_save_path = actual_output_dir / "rpv.json"
            with open(self.rpv_save_path, "w") as f:
                json.dump(self.rpv, f)
        
        if self.tau_and_cpd is not None:
            self.tau_and_cpd_save_path = actual_output_dir / "tau_cpd.json"
            with open(self.tau_and_cpd_save_path, "w") as f:
                json.dump(self.tau_and_cpd, f)

        # Save metadata
        metadata_save_path = actual_output_dir / "metadata.json"
        metadata = {
            **self.src_frame.kwargs,
        }
        with open(metadata_save_path, "w") as f:
            json.dump(metadata, f)
            

    def _rpv_text(self, rpv: np.ndarray):
        theta, phi, psi, tx, ty, tz = rpv
        return [
            "pitch up" if theta > 0 else "pitch down",
            "yaw right" if phi > 0 else "yaw left",
            "roll clockwise" if psi > 0 else "roll counterclockwise",
            "translate right" if tx > 0 else "translate left",
            "translate down" if ty > 0 else "translate up",
            "translate forward" if tz > 0 else "translate backward",
        ]
    
    def cal_rpv(self) -> Dict[str, List]:
        """
        Calculate the relative pose vector (RPV) from source frame to target frame.
        """
        ### core computation
        self.rpose = np.linalg.inv(self.src_frame.pose) @ self.tgt_frame.pose
        self.rR = self.rpose[:3, :3]
        self.rT = self.rpose[:3, 3]

        theta = np.degrees(np.arctan2(self.rR[2, 1], self.rR[2, 2]))
        phi = np.degrees(np.arcsin(-self.rR[2, 0]))
        psi = np.degrees(np.arctan2(self.rR[1, 0], self.rR[0, 0]))
        v = np.array([theta, phi, psi, self.rT[0], self.rT[1], self.rT[2]])
        self.rpv = {
            "value": v.tolist(),
            "text": self._rpv_text(v),
        }
        return self.rpv

    def _get_angle(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        """
        (x, y, z)_1
        (x, y, z)_2
        (x, y, z)_3

        output:
            angle ABC (if one of the side is too small, return 0.)
        """
        BA = A - B
        BC = C - B

        norm_BA = np.linalg.norm(BA)
        norm_BC = np.linalg.norm(BC)

        if norm_BA < 1e-8 or norm_BC < 1e-8:
            # print("⚠️⚠️ Warning: One of the vectors has near-zero magnitude, return angle as 0.")
            return 0.

        BA = BA / norm_BA
        BC = BC / norm_BC

        angle = np.arccos(np.dot(BA, BC)) * 180 / np.pi
        return angle

    def _unproject(self, pose, K_inv, pixelx, pixely, depth) -> np.ndarray:
        """
        (x, y) -> (x, y, z)_wrd
        """
        point3d_camera_coords = depth * K_inv @ np.array([pixelx, pixely, 1])
        point3D_world = pose[:3, :3] @ point3d_camera_coords + pose[:3, 3] # vec_wrd = P^cam_wrd * vec_cam
        return point3D_world

    def _reproject(self, pose, K, point3D) -> np.ndarray:
        """
        (x, y, z)_wrd -> (x, y)
        """
        P = K @ np.linalg.inv(pose)[:3]
        projection = P[:3, :3] @ point3D + P[:3, 3]
        projection[:2] /= projection[2:] # normalize
        return projection[:2]

    def cal_tau_and_cpd(self) -> Dict[str, float]:
        """
        return tau (in degrees) and central point deviation (cpd) (in pixels)
        """
        ### pre-request
        if self.src_frame.K is None or self.tgt_frame.K is None:
            raise ValueError("Intrinsic matrix K is required for both source and target frames to compute tau and cpd.")
        if self.src_frame.depth is None or self.tgt_frame.depth is None:
            raise ValueError("Depth images are required for both source and target frames to compute tau and cpd.")

        # unproject, from 2D to 3D World Coordinate
        central_pixel_0_x = self.src_frame.color.size[1] // 2
        central_pixel_0_y = self.src_frame.color.size[0] // 2
        central_pixel_0_depth = np.array(self.src_frame.depth)[central_pixel_0_y, central_pixel_0_x] / 1000
        centeral_pixel_3D_point_0_world = self._unproject(self.src_frame.pose, self.src_frame.K_inv, central_pixel_0_x, central_pixel_0_y, central_pixel_0_depth)

        # unproject, from 2D to 3D World Coordinate
        central_pixel_100_x = self.tgt_frame.color.size[1] // 2
        central_pixel_100_y = self.tgt_frame.color.size[0] // 2
        central_pixel_100_depth = np.array(self.tgt_frame.depth)[central_pixel_100_y, central_pixel_100_x] / 1000
        centeral_pixel_3D_point_100_world = self._unproject(self.tgt_frame.pose, self.tgt_frame.K_inv, central_pixel_100_x, central_pixel_100_y, central_pixel_100_depth)

        # reproject, from 3D World Coordinate to 2D
        reprojection_0_to_100 = self._reproject(self.tgt_frame.pose, self.tgt_frame.K, centeral_pixel_3D_point_0_world)
        reprojection_100_to_0 = self._reproject(self.src_frame.pose, self.src_frame.K, centeral_pixel_3D_point_100_world)

        center0_world = self.src_frame.pose[:3, 3]
        center100_world = self.tgt_frame.pose[:3, 3]

        angle_point0 = self._get_angle(
            center0_world,
            centeral_pixel_3D_point_0_world,
            center100_world
        )
        angle_point100 = self._get_angle(
            center0_world,
            centeral_pixel_3D_point_100_world,
            center100_world
        )

        distance_0 = np.linalg.norm(np.array((central_pixel_0_x, central_pixel_0_y)) - reprojection_100_to_0)
        distance_100 = np.linalg.norm(np.array((central_pixel_100_x, central_pixel_100_y)) - reprojection_0_to_100) # in pixel scale

        self.tau_and_cpd = {
            "tau": float(angle_point0),
            "cpd": float(distance_0),
        }
        return self.tau_and_cpd
