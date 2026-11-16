//! # threestudio-rs
//!
//! A Rust port scaffold of [threestudio](https://github.com/threestudio-project/threestudio)
//! built on [Burn](https://burn.dev) and [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! The module layout mirrors the Python package:
//!
//! | threestudio (Python)              | threestudio-rs           |
//! |-----------------------------------|--------------------------|
//! | `threestudio/utils/config.py`     | [`config`]               |
//! | `threestudio/data/uncond.py`      | [`data`]                 |
//! | `threestudio/models/geometry`     | [`models::geometry`]     |
//! | `threestudio/models/networks.py`  | [`models::encoding`]     |
//! | `threestudio/models/renderers`    | [`models::renderer`]     |
//! | `threestudio/models/guidance`     | [`models::guidance`]     |
//! | `threestudio/systems`             | [`systems`]              |
//! | `launch.py`                       | `launch` binary          |

pub mod backend;
pub mod config;
pub mod data;
pub mod math;
pub mod models;
pub mod systems;
