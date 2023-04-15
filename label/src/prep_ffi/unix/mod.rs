use crate::prep_ffi::{make_labels, unify, LIVER_HEIGHT, LIVER_SIZE, LIVER_WIDTH};
use ndarray::Array3;
use std::ffi::{c_char, CStr, OsStr};
use std::path::Path;

#[repr(i32)]
pub enum InvokeResult {
    Ok = 0,
    OpenError = 1,
    ShapeError = 2,
    SaveError = 3,
    CreateDirError = 4,
}

#[inline]
fn cstr_to_rust_path(s: &CStr) -> &Path {
    use std::os::unix::prelude::OsStrExt;

    let os_fmt = OsStr::from_bytes(s.to_bytes());
    os_fmt.as_ref()
}

/// Preprocess 3D liver image stored in `.npy` file.
///
/// # Safety
///
/// The caller must ensure that:
///
/// - `npy_path` and `dest_path` are C-style strings.
/// - The `.npy` file has the shape like \[z, 512, 512\] and data type of byte(u8).
///
/// The behavior is undefined if any of the requirements above is not met.
#[no_mangle]
pub unsafe fn unify_liver_area(npy_path: *const c_char, dest_path: *const c_char) -> InvokeResult {
    let npy_path = CStr::from_ptr(npy_path);
    let npy_path = cstr_to_rust_path(npy_path);
    // [Z-axis, Height, Width]
    let liver_3d: Array3<u8> = match ndarray_npy::read_npy(npy_path) {
        Ok(v) => v,
        _ => return InvokeResult::OpenError,
    };
    let mut liver_flattened = liver_3d.into_raw_vec();
    if liver_flattened.len() % LIVER_SIZE != 0 {
        return InvokeResult::ShapeError;
    }

    unify(liver_flattened.as_mut()); // core algorithm

    let z_len = liver_flattened.len() / LIVER_SIZE;
    let array =
        match Array3::<u8>::from_shape_vec((z_len, LIVER_HEIGHT, LIVER_WIDTH), liver_flattened) {
            Ok(arr) => arr,
            _ => return InvokeResult::ShapeError,
        };
    let dest_path = CStr::from_ptr(dest_path);
    let dest_path = cstr_to_rust_path(dest_path);
    match ndarray_npy::write_npy(dest_path, &array) {
        Ok(_) => InvokeResult::Ok,
        Err(_) => InvokeResult::SaveError,
    }
}

/// Generating label of file `.npy` in json format.
///
/// # Safety
///
/// The caller must ensure that both `npy_path` and `dest_path` are C-style string.
/// The behavior is undefined if the requirement is not met.
#[no_mangle]
pub unsafe fn make_sequential_labels(
    npy_path: *const c_char,
    dest_path: *const c_char,
) -> InvokeResult {
    let npy_path = CStr::from_ptr(npy_path);
    let npy_path = cstr_to_rust_path(npy_path);
    // [Z-axis, Height, Width]
    let liver_3d: Array3<u8> = match ndarray_npy::read_npy(npy_path) {
        Ok(v) => v,
        _ => return InvokeResult::OpenError,
    };
    let mut liver_flattened = liver_3d.into_raw_vec();
    if liver_flattened.len() % LIVER_SIZE != 0 {
        return InvokeResult::ShapeError;
    }

    let dest_path = CStr::from_ptr(dest_path);
    let dest_path = cstr_to_rust_path(dest_path).to_path_buf();
    if std::fs::create_dir_all(dest_path.as_path()).is_err() {
        return InvokeResult::CreateDirError;
    }
    make_labels(liver_flattened.as_mut_slice(), dest_path);
    InvokeResult::Ok
}
