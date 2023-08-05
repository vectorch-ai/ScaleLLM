// A simple C wrapper of hf-tokenzier library
// ported from https://github.com/mlc-ai/tokenizers-cpp

// Import the needed libraries
use std::ffi::CStr;
use std::os::raw::c_char;
use tokenizers::tokenizer::Tokenizer;

pub struct TokenizerWrapper {
    // The tokenizer
    tokenizer: Tokenizer,
    // Holds the encoded ids to avoid dropping them
    encode_ids: Vec<u32>,
    // Holds the decoded string to avoid dropping it
    decode_str: String,
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
        self.decode_str = self.tokenizer.decode(ids, skip_special_tokens).unwrap();
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C string to Rust string"),
    };

    let boxed = Box::new(TokenizerWrapper {
        tokenizer: Tokenizer::from_file(path_str).unwrap().into(),
        encode_ids: Vec::new(),
        decode_str: String::new(),
    });

    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tokenizer_from_pretrained(identifier: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(identifier) };
    let identifier_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C string to Rust string"),
    };

    let boxed = Box::new(TokenizerWrapper {
        tokenizer: Tokenizer::from_pretrained(identifier_str, None).unwrap().into(),
        encode_ids: Vec::new(),
        decode_str: String::new(),
    });

    Box::into_raw(boxed)
}

#[no_mangle]
pub extern "C" fn tokenizer_encode(
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
pub extern "C" fn tokenizer_get_encode_ids(
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
pub extern "C" fn tokenizer_decode(
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
pub extern "C" fn tokenizer_get_decode_str(
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
pub extern "C" fn tokenizer_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}
