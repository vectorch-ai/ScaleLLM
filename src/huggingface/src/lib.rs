// A simple C wrapper of safetensors and tokenizers library
// adapted from https://github.com/huggingface/safetensors/tree/c_bindings

// Import the needed libraries
use core::ffi::c_uint;
use core::str::Utf8Error;
use safetensors::tensor::{SafeTensorError, SafeTensors};
use safetensors::Dtype as RDtype;
use std::ffi::{c_char, CStr, CString};
use std::mem::forget;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;

// Status codes should be in sync with the ones defined in `tensor.rs`
// https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L15
#[repr(C)]
pub enum Status {
    NullPointer = -2,
    Utf8Error,
    Ok,
    InvalidHeader,
    InvalidHeaderStart,
    InvalidHeaderDeserialization,
    HeaderTooLarge,
    HeaderTooSmall,
    InvalidHeaderLength,
    TensorNotFound,
    TensorInvalidInfo,
    InvalidOffset,
    IoError,
    JsonError,
    InvalidTensorView,
    MetadataIncompleteBuffer,
    ValidationOverflow,
}

#[derive(Debug, Error)]
#[repr(C)]
enum CError {
    #[error("{0}")]
    NullPointer(String),

    #[error("{0}")]
    Utf8Error(#[from] Utf8Error),

    #[error("{0}")]
    SafeTensorError(#[from] SafeTensorError),
}

impl Into<Status> for CError {
    fn into(self) -> Status {
        match self {
            CError::NullPointer(_) => Status::NullPointer,
            CError::Utf8Error(_) => Status::Utf8Error,
            CError::SafeTensorError(err) => match err {
                SafeTensorError::InvalidHeader => Status::InvalidHeader,
                SafeTensorError::InvalidHeaderStart => Status::InvalidHeaderStart,
                SafeTensorError::InvalidHeaderDeserialization => {
                    Status::InvalidHeaderDeserialization
                }
                SafeTensorError::HeaderTooLarge => Status::HeaderTooLarge,
                SafeTensorError::HeaderTooSmall => Status::HeaderTooSmall,
                SafeTensorError::InvalidHeaderLength => Status::InvalidHeaderLength,
                SafeTensorError::TensorNotFound(_) => Status::TensorNotFound,
                SafeTensorError::TensorInvalidInfo => Status::TensorInvalidInfo,
                SafeTensorError::InvalidOffset(_) => Status::InvalidOffset,
                SafeTensorError::IoError(_) => Status::IoError,
                SafeTensorError::JsonError(_) => Status::JsonError,
                SafeTensorError::InvalidTensorView(_, _, _) => Status::InvalidTensorView,
                SafeTensorError::MetadataIncompleteBuffer => Status::MetadataIncompleteBuffer,
                SafeTensorError::ValidationOverflow => Status::ValidationOverflow,
            },
        }
    }
}

/// The various available dtypes. They MUST be in increasing alignment order
// the Dtype has to be in sync with the one defined in `tensor.rs`
// https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L631
#[repr(C)]
pub enum Dtype {
    /// Boolan type
    BOOL,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
}

impl Dtype {
    fn to_dtype(self) -> RDtype {
        match self {
            Dtype::BOOL => RDtype::BOOL,
            Dtype::U8 => RDtype::U8,
            Dtype::I8 => RDtype::I8,
            Dtype::I16 => RDtype::I16,
            Dtype::U16 => RDtype::U16,
            Dtype::I32 => RDtype::I32,
            Dtype::U32 => RDtype::U32,
            Dtype::I64 => RDtype::I64,
            Dtype::U64 => RDtype::U64,
            Dtype::F16 => RDtype::F16,
            Dtype::BF16 => RDtype::BF16,
            Dtype::F32 => RDtype::F32,
            Dtype::F64 => RDtype::F64,
        }
    }
}

impl From<RDtype> for Dtype {
    fn from(dtype: RDtype) -> Dtype {
        match dtype {
            RDtype::BOOL => Dtype::BOOL,
            RDtype::U8 => Dtype::U8,
            RDtype::I8 => Dtype::I8,
            RDtype::I16 => Dtype::I16,
            RDtype::U16 => Dtype::U16,
            RDtype::I32 => Dtype::I32,
            RDtype::U32 => Dtype::U32,
            RDtype::I64 => Dtype::I64,
            RDtype::U64 => Dtype::U64,
            RDtype::F16 => Dtype::F16,
            RDtype::BF16 => Dtype::BF16,
            RDtype::F32 => Dtype::F32,
            RDtype::F64 => Dtype::F64,
            d => panic!("Unhandled dtype {d:?}"),
        }
    }
}

pub struct Handle {
    safetensors: SafeTensors<'static>,
    buffer: *const u8,
}

#[repr(C)]
pub struct View {
    dtype: Dtype,
    rank: usize,
    shape: *const usize,
    start: usize,
    stop: usize,
}

/// Attempt to deserialize the content of `buffer`, reading `buffer_len` bytes as a safentesors
/// data buffer.
///
/// # Arguments
///
/// * `handle`: In-Out pointer to store the resulting safetensors reference is sucessfully deserialized
/// * `buffer`: Buffer to attempt to read data from
/// * `buffer_len`: Number of bytes we can safely read from the deserialize the safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub extern "C" fn safetensors_deserialize(
    handle: *mut *mut Handle,
    buffer: *const u8,
    buffer_len: usize,
) -> Status {
    match unsafe { _deserialize(buffer, buffer_len) } {
        Ok(safetensors) => unsafe {
            let heap_handle = Box::new(Handle {
                safetensors,
                buffer,
            });
            let raw = Box::into_raw(heap_handle);
            handle.write(raw);

            Status::Ok
        },
        Err(err) => err.into(),
    }
}

/// Free the resources hold by the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer ot the safetensors we want to release the resources of
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub unsafe extern "C" fn safetensors_destroy(handle: *mut Handle) -> Status {
    if !handle.is_null() {
        // Restore the heap allocated handle and explicitly drop it
        drop(Box::from_raw(handle));
    }

    Status::Ok
}

/// Retrieve the list of tensor's names currently stored in the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to query tensor's names from
/// * `ptr`: In-Out pointer to store the array of strings representing all the tensor's names
/// * `len`: Number of strings stored in `ptr`
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub unsafe extern "C" fn safetensors_names(
    handle: *const Handle,
    ptr: *mut *const *const c_char,
    len: *mut c_uint,
) -> Status {
    let names = (*handle).safetensors.names();

    // We need to convert the String repr to a C-friendly repr (NUL terminated)
    let c_names = names
        .into_iter()
        .map(|name| {
            // Nul-terminated string
            let s = CString::from_vec_unchecked(name.clone().into_bytes());
            let ptr = s.as_ptr();

            // Advise Rust we will take care of the desallocation (see `safetensors_free_names`)
            forget(s);

            ptr
        })
        .collect::<Vec<_>>();

    unsafe {
        ptr.write(c_names.as_ptr());
        len.write(c_names.len() as c_uint);

        forget(c_names);

        Status::Ok
    }
}

/// Free the resources used to represent the list of tensor's names stored in the safetensors.
/// This must follow any call to `safetensors_names()` to clean up underlying resources.
///
/// # Arguments
///
/// * `names`: Pointer to the array of strings we want to release resources of
/// * `len`: Number of strings hold by `names` array
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub extern "C" fn safetensors_free_names(names: *const *const c_char, len: c_uint) -> Status {
    let len = len as usize;

    unsafe {
        // Get back our vector.
        let v = Vec::from_raw_parts(names.cast_mut(), len, len);

        // Now drop all the string.
        for elem in v {
            let _ = CString::from_raw(elem.cast_mut());
        }
    }

    Status::Ok
}

/// Return the number of tensors stored in this safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to know the number of tensors of.
///
/// returns: usize Number of tensors in the safetensors
#[no_mangle]
pub unsafe extern "C" fn safetensors_num_tensors(handle: *const Handle) -> usize {
    (*handle).safetensors.len()
}

/// Return the number of bytes required to represent a single element from the specified dtype
///
/// # Arguments
///
/// * `dtype`: The data type we want to know the number of bytes required
///
/// returns: usize Number of bytes for this specific `dtype`
#[no_mangle]
pub extern "C" fn safetensors_dtype_size(dtype: Dtype) -> usize {
    dtype.to_dtype().size()
}

/// Attempt to retrieve the metadata and content for the tensor associated with `name` storing the
/// result to the memory location pointed by `view` pointer.
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to retrieve the tensor from.
/// * `view`: In-Out pointer to store the tensor if successfully found to belong to the safetensors
/// * `name`: The name of the tensor to retrieve from the safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error what caught up
#[no_mangle]
pub extern "C" fn safetensors_get_tensor(
    handle: *const Handle,
    view: *mut *mut View,
    name: *const c_char,
) -> Status {
    match unsafe { _get_tensor(handle, view, name) } {
        Ok(_) => Status::Ok,
        Err(err) => err.into(),
    }
}

/// Free the resources used by a TensorView to expose metadata + content to the C-FFI layer
///
/// # Arguments
///
/// * `ptr`: Pointer to the TensorView we want to release the underlying resources of
///
/// returns: `Status::Ok = 0` if resources were successfully freed
#[no_mangle]
pub extern "C" fn safetensors_free_tensor(ptr: *mut View) -> Status {
    unsafe {
        // Restore the heap allocated view and explicitly drop it
        drop(Box::from_raw(ptr));

        Status::Ok
    }
}

/// Deserialize the content pointed by `buffer`, reading `buffer_len` number of bytes from it
///
/// # Arguments
///
/// * `buffer`: The raw buffer to read from
/// * `buffer_len`: The number of bytes to safely read from the `buffer`
///
/// returns: Result<SafeTensors, CError>
#[inline(always)]
unsafe fn _deserialize(
    buffer: *const u8,
    buffer_len: usize,
) -> Result<SafeTensors<'static>, CError> {
    if buffer.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `buffer` when accessing deserialize".to_string(),
        ));
    }

    let data = unsafe { std::slice::from_raw_parts(buffer, buffer_len) };
    SafeTensors::deserialize(&data).map_err(|err| CError::SafeTensorError(err))
}

/// Retrieve a tensor from the underlying safetensors pointed by `handle` and referenced by it's `name`.
/// If found, the resulting view will populate the memory location pointed by `ptr`
///
/// # Arguments
///
/// * `handle`: Handle to the underlying safetensors we want to retrieve the tensor from
/// * `ptr`: The in-out pointer to populate if the tensor is found
/// * `name`: The name of the tensor we want to retrieve
///
/// returns: Result<(), CError>
unsafe fn _get_tensor(
    handle: *const Handle,
    ptr: *mut *mut View,
    name: *const c_char,
) -> Result<(), CError> {
    if name.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `name` when accessing get_tensor".to_string(),
        ));
    }
    if handle.is_null() {
        return Err(CError::NullPointer(
            "Null pointer `handle` when accessing get_tensor".to_string(),
        ));
    }
    let name = CStr::from_ptr(name).to_str()?;
    let st_view = (*handle).safetensors.tensor(name)?;

    let start_ptr = (*handle).buffer as usize;

    unsafe {
        let shape = st_view.shape().to_vec();
        let data = st_view.data();
        let start = data.as_ptr() as usize - start_ptr;
        let stop = start + data.len();

        let view = Box::new(View {
            dtype: st_view.dtype().into(),
            rank: shape.len(),
            shape: shape.as_ptr(),
            start,
            stop,
        });
        forget(shape);

        ptr.write(Box::into_raw(view));
    }

    Ok(())
}


// A simple C wrapper of hf-tokenzier library
// ported from https://github.com/mlc-ai/tokenizers-cpp

pub struct TokenizerWrapper {
    // The tokenizer
    tokenizer: Tokenizer,
    // Holds the encoded ids to avoid dropping them
    encode_ids: Vec<u32>,
    // Holds the decoded string to avoid dropping it
    decode_str: String,
    // Holds the result of the token_to_id function
    id_to_token_result: String,
}

impl TokenizerWrapper {
    pub fn encode(&mut self, text: &str, add_special_tokens: bool) {
        // Encode the text and store the ids
        self.encode_ids = Vec::from(
            self.tokenizer
                .encode(text, add_special_tokens)
                .unwrap()
                .get_ids(),
        );
    }

    pub fn decode(&mut self, ids: Vec<u32>, skip_special_tokens: bool) {
        // Decode the ids and store the string
        self.decode_str = self.tokenizer.decode(&ids, skip_special_tokens).unwrap();
    }

    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }
}

#[no_mangle]
extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C string to Rust string"),
    };

    let boxed = Box::new(TokenizerWrapper {
        tokenizer: Tokenizer::from_file(path_str).unwrap().into(),
        encode_ids: Vec::new(),
        decode_str: String::new(),
        id_to_token_result: String::new(),
    });

    Box::into_raw(boxed)
}

#[no_mangle]
extern "C" fn tokenizer_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: bool,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        (*handle).encode(input_data, add_special_tokens);
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_encode_ids(
    handle: *mut TokenizerWrapper,
    out_data: *mut *mut u32,
    out_len: *mut usize,
) {
    unsafe {
        *out_data = (*handle).encode_ids.as_mut_ptr();
        *out_len = (*handle).encode_ids.len()
    }
}

#[no_mangle]
extern "C" fn tokenizer_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
) {
    unsafe {
        let input_data = Vec::from(std::slice::from_raw_parts(input_ids, len));
        (*handle).decode(input_data, skip_special_tokens);
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (*handle).decode_str.len();
    }
}

#[no_mangle]
extern "C" fn tokenizer_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}

#[no_mangle]
extern "C" fn tokenizer_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize
) {
    unsafe {
        let token: &str = std::str::from_utf8(std::slice::from_raw_parts(token, len)).unwrap();
        let id = (*handle).tokenizer.token_to_id(token);
        match id {
            Some(id) => id as i32,
            None => -1,
        };
    }
}

#[no_mangle]
extern "C" fn tokenizer_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let str = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = match str {
            Some(s) => s,
            None => String::from(""),
        };

        *out_cstr = (*handle).id_to_token_result.as_mut_ptr();
        *out_len = (*handle).id_to_token_result.len();
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_vocab_size(
    handle: *mut TokenizerWrapper, 
    with_added_tokens: bool) -> usize {
    unsafe {
        (*handle).get_vocab_size(with_added_tokens)
    }
}
