//! Random camera sampling and ray generation.
//!
//! Port of `threestudio/data/uncond.py` (`RandomCameraIterableDataset`) and
//! `threestudio/utils/ops.py` (`get_ray_directions`, `get_rays`).

use crate::config::RandomCameraConfig;
use crate::math::{self, Vec3};
use burn::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Camera-to-world transform: columns are right, up, back (-lookat); `position` is the eye.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub right: Vec3,
    pub up: Vec3,
    pub back: Vec3,
    pub position: Vec3,
    pub fovy_deg: f32,
    pub elevation_deg: f32,
    pub azimuth_deg: f32,
    pub distance: f32,
}

impl Camera {
    /// Spherical camera looking at the origin with +z up (threestudio's convention).
    pub fn look_at_origin(
        elevation_deg: f32,
        azimuth_deg: f32,
        distance: f32,
        fovy_deg: f32,
    ) -> Self {
        let (e, a) = (elevation_deg.to_radians(), azimuth_deg.to_radians());
        let position = [
            distance * e.cos() * a.cos(),
            distance * e.cos() * a.sin(),
            distance * e.sin(),
        ];
        let world_up = [0.0, 0.0, 1.0];
        let lookat = math::normalize(math::sub([0.0; 3], position));
        let mut right = math::cross(lookat, world_up);
        if math::norm(right) < 1e-6 {
            // Looking straight up/down: pick an arbitrary right vector.
            right = [0.0, 1.0, 0.0];
        }
        let right = math::normalize(right);
        let up = math::normalize(math::cross(right, lookat));
        Self {
            right,
            up,
            back: math::scale(lookat, -1.0),
            position,
            fovy_deg,
            elevation_deg,
            azimuth_deg,
            distance,
        }
    }

    pub fn focal_length(&self, height: usize) -> f32 {
        0.5 * height as f32 / (0.5 * self.fovy_deg.to_radians()).tan()
    }

    fn rotate(&self, d: Vec3) -> Vec3 {
        math::add(
            math::add(math::scale(self.right, d[0]), math::scale(self.up, d[1])),
            math::scale(self.back, d[2]),
        )
    }
}

/// Camera-space ray directions for every pixel, like `get_ray_directions`
/// (OpenGL convention: x right, y up, camera looks down -z).
pub fn ray_directions(height: usize, width: usize, focal: f32) -> Vec<Vec3> {
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let mut dirs = Vec::with_capacity(height * width);
    for j in 0..height {
        for i in 0..width {
            let x = i as f32 + 0.5;
            let y = j as f32 + 0.5;
            dirs.push([(x - cx) / focal, -(y - cy) / focal, -1.0]);
        }
    }
    dirs
}

/// A batch of rays from one camera, flattened row-major over pixels.
#[derive(Debug, Clone)]
pub struct RayBatch {
    pub camera: Camera,
    pub height: usize,
    pub width: usize,
    pub rays_o: Vec<Vec3>,
    pub rays_d: Vec<Vec3>,
}

impl RayBatch {
    pub fn from_camera(camera: Camera, height: usize, width: usize) -> Self {
        let dirs = ray_directions(height, width, camera.focal_length(height));
        let rays_d: Vec<Vec3> = dirs
            .into_iter()
            .map(|d| math::normalize(camera.rotate(d)))
            .collect();
        let rays_o = vec![camera.position; rays_d.len()];
        Self {
            camera,
            height,
            width,
            rays_o,
            rays_d,
        }
    }

    pub fn len(&self) -> usize {
        self.rays_d.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rays_d.is_empty()
    }

    /// `(rays_o, rays_d)` as `[n_rays, 3]` tensors.
    pub fn to_tensors<B: Backend>(&self, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let n = self.len();
        let flat = |v: &[Vec3]| v.iter().flatten().copied().collect::<Vec<f32>>();
        (
            Tensor::from_data(TensorData::new(flat(&self.rays_o), [n, 3]), device),
            Tensor::from_data(TensorData::new(flat(&self.rays_d), [n, 3]), device),
        )
    }
}

/// Infinite random-camera "dataset", equivalent to `RandomCameraIterableDataset`.
pub struct RandomCameraDataset {
    cfg: RandomCameraConfig,
    rng: StdRng,
}

impl RandomCameraDataset {
    pub fn new(cfg: RandomCameraConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn uniform(&mut self, r: [f32; 2]) -> f32 {
        if r[1] > r[0] {
            self.rng.random_range(r[0]..r[1])
        } else {
            r[0]
        }
    }

    pub fn sample_camera(&mut self) -> Camera {
        let elev = self.uniform(self.cfg.elevation_range);
        let azim = self.uniform(self.cfg.azimuth_range);
        let dist = self.uniform(self.cfg.camera_distance_range);
        let fovy = self.uniform(self.cfg.fovy_range);
        Camera::look_at_origin(elev, azim, dist, fovy)
    }

    pub fn sample(&mut self) -> RayBatch {
        let cam = self.sample_camera();
        RayBatch::from_camera(cam, self.cfg.height, self.cfg.width)
    }

    /// Deterministic 360° evaluation orbit, like `RandomCameraDataset` in val/test mode.
    pub fn eval_orbit(&self, n_views: usize, elevation_deg: f32) -> Vec<RayBatch> {
        (0..n_views)
            .map(|k| {
                let azim = -180.0 + 360.0 * k as f32 / n_views as f32;
                let cam = Camera::look_at_origin(
                    elevation_deg,
                    azim,
                    self.cfg.eval_camera_distance,
                    self.cfg.eval_fovy_deg,
                );
                RayBatch::from_camera(cam, self.cfg.height, self.cfg.width)
            })
            .collect()
    }
}

impl Iterator for RandomCameraDataset {
    type Item = RayBatch;

    fn next(&mut self) -> Option<RayBatch> {
        Some(self.sample())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_frame_is_orthonormal_and_faces_origin() {
        let cam = Camera::look_at_origin(30.0, 45.0, 2.0, 60.0);
        for v in [cam.right, cam.up, cam.back] {
            assert!((math::norm(v) - 1.0).abs() < 1e-5);
        }
        assert!(math::dot(cam.right, cam.up).abs() < 1e-5);
        assert!(math::dot(cam.up, cam.back).abs() < 1e-5);
        assert!((math::norm(cam.position) - 2.0).abs() < 1e-5);
        // back points from the origin to the eye
        let to_eye = math::normalize(cam.position);
        assert!((math::dot(to_eye, cam.back) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rays_are_unit_and_center_ray_hits_origin() {
        let cam = Camera::look_at_origin(10.0, -70.0, 1.5, 50.0);
        let batch = RayBatch::from_camera(cam, 9, 9);
        assert_eq!(batch.len(), 81);
        for d in &batch.rays_d {
            assert!((math::norm(*d) - 1.0).abs() < 1e-5);
        }
        // center pixel (4,4) of an odd image points straight at the origin
        let c = batch.rays_d[4 * 9 + 4];
        let expected = math::normalize(math::scale(cam.position, -1.0));
        assert!((math::dot(c, expected) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ray_directions_follow_opengl_convention() {
        let d = ray_directions(2, 2, 1.0);
        assert_eq!(d.len(), 4);
        // top-left pixel: x < 0, y > 0, z = -1
        assert!(d[0][0] < 0.0 && d[0][1] > 0.0 && d[0][2] == -1.0);
    }

    #[test]
    fn random_cameras_respect_ranges() {
        let cfg = RandomCameraConfig::default();
        let mut ds = RandomCameraDataset::new(cfg.clone(), 42);
        for _ in 0..100 {
            let c = ds.sample_camera();
            assert!(
                c.elevation_deg >= cfg.elevation_range[0]
                    && c.elevation_deg <= cfg.elevation_range[1]
            );
            assert!(
                c.distance >= cfg.camera_distance_range[0]
                    && c.distance <= cfg.camera_distance_range[1]
            );
        }
    }

    #[test]
    fn tensors_have_expected_shape() {
        let batch = RayBatch::from_camera(Camera::look_at_origin(0.0, 0.0, 1.0, 60.0), 4, 6);
        let (o, d) = batch.to_tensors::<crate::backend::Cpu>(&Default::default());
        assert_eq!(o.dims(), [24, 3]);
        assert_eq!(d.dims(), [24, 3]);
    }
}
