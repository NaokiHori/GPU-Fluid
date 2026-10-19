/*
 * Copyright 2022 Naoki Hori
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// https://github.com/NaokiHori/SimpleNpyIO

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include "snpyio.h"

#if 8 != CHAR_BIT
#error "CHAR_BIT is not 8"
#endif

// all npy files should start from this magic string
static const char magic_string[] = {"\x93NUMPY"};

// header size is divisible by this number
static const size_t header_block_size = 64;

// null character
static const char null_char = '\0';

// general messages for logging and error handling
#define SNPYIO_MESSAGE(stream, level, ...){    \
  fprintf(stream, "[NPYIO %s]\n", level);      \
  fprintf(stream, "    line: %d\n", __LINE__); \
  fprintf(stream, "    func: %s\n", __func__); \
  fprintf(stream, "    ");                     \
  fprintf(stream, __VA_ARGS__);                \
  fflush(stream);                              \
}

// logger
#if defined(SNPYIO_ENABLE_LOGGING)
#define SNPYIO_LOGGING(...){              \
  FILE * fp = fopen("snpyio.log", "a");   \
  SNPYIO_MESSAGE(fp, "LOG", __VA_ARGS__); \
  fclose(fp);                             \
}
#else
#define SNPYIO_LOGGING(...)
#endif

// normal error
#define SNPYIO_ERROR(...){                      \
  SNPYIO_MESSAGE(stderr, "ERROR", __VA_ARGS__); \
}

// fatal error
#define SNPYIO_FATAL(...){                      \
  SNPYIO_MESSAGE(stderr, "FATAL", __VA_ARGS__); \
  exit(EXIT_FAILURE);                           \
}

// general-purpose singly-linked list
// by default used by the memory manager and the parser
typedef struct node_t_ {
  void * ptr;
  struct node_t_ * next;
} node_t;

// memory pool storing all pointers to the allocated buffers in this library
static node_t * memory = NULL;

// kernel memory allocator
static void * kernel_alloc(
    const size_t count,
    const size_t size
){
  if(SIZE_MAX / size < count){
    SNPYIO_FATAL("request too large memory (%zu, %zu)\n", count, size);
  }
  void * ptr = malloc(count * size);
  if(NULL == ptr){
    SNPYIO_FATAL("memory allocation failed: (%zu, %zu)\n", count, size);
  }
  return ptr;
}

// kernel memory deallocator
static void kernel_free(
    void * ptr
){
  free(ptr);
}

// general memory allocator
static void * memory_alloc(
    const size_t count,
    const size_t size
){
  void * ptr = kernel_alloc(count, size);
  node_t * node = kernel_alloc(1, sizeof(node_t));
  node->ptr = ptr;
  node->next = memory;
  memory = node;
  return ptr;
}

// deallocate a node holding the given pointer
static int detach_list(
    const void * ptr
){
  if(NULL == ptr){
    return 1;
  }
  node_t ** node = &memory;
  while(*node){
    if(ptr == (*node)->ptr){
      // keep next node
      node_t * node_next = (*node)->next;
      // clean-up current node
      // NOTE: ptr is NOT freed while the node is freed
      kernel_free(*node);
      // update connection
      *node = node_next;
      return 0;
    }
    node = &((*node)->next);
  }
  return 1;
}

// general memory deallocator
static void memory_free(
    void * ptr
){
  detach_list(ptr);
  kernel_free(ptr);
}

// free all buffers attached to the memory pool
static void error_handlings(
    void
){
  while(memory){
    node_t * next = memory->next;
    kernel_free(memory->ptr);
    kernel_free(memory);
    memory = next;
  }
}

// check this architecture is big-endian
static bool is_big_endian(
    void
){
  // big    endian: 0x1 0x0
  // little endian: 0x0 0x1
  const uint16_t val16 = 1 << 8;
  uint8_t vals8[] = {0, 0};
  memcpy(vals8, &val16, 2 * sizeof(uint8_t));
  if(0 != vals8[0]){
    return true;
  }else{
    return false;
  }
}

// perform endian swap
static int convert_endian(
    const size_t size,
    void * val
){
  const size_t nitems = size / sizeof(uint8_t);
  uint8_t * buf0 = memory_alloc(nitems, sizeof(uint8_t));
  uint8_t * buf1 = memory_alloc(nitems, sizeof(uint8_t));
  // copy
  memcpy(buf0, val, size);
  // swap
  for(size_t n = 0; n < nitems; n++){
    buf1[n] = buf0[nitems - n - 1];
  }
  // copy back
  memcpy(val, buf1, size);
  memory_free(buf0);
  memory_free(buf1);
  return 0;
}

// fread wrapper
#define MYFREAD( \
    ptr, \
    size, \
    nitems, \
    stream, \
    error_handler \
){ \
  if(nitems != fread(ptr, size, nitems, stream)){ \
    SNPYIO_ERROR("failed to read from file\n"); \
    error_handler; \
  } \
}

// reject NULL file stream
static int sanitise_fp(
    const FILE * fp
){
  if(NULL == fp){
    SNPYIO_ERROR("fp is NULL, check file is properly opened\n");
    return 1;
  }
  return 0;
}

// check if this file starts with a magic string
static int load_magic_string(
    FILE * fp,
    size_t * header_size
){
  const size_t nitems = strlen(magic_string);
  // allocate buffer and load from file
  uint8_t * buf = memory_alloc(nitems, sizeof(uint8_t));
  MYFREAD(buf, sizeof(uint8_t), nitems, fp, return 1);
  const size_t buf_size = nitems * sizeof(uint8_t);
  if(0 != memcmp(buf, magic_string, buf_size)){
    SNPYIO_ERROR("file does not begin with \"\\x93NUMPY\"\n");
    return 1;
  }
  memory_free(buf);
  *header_size += buf_size;
  return 0;
}

// load and check major/minor versions
static int load_versions(
    FILE * fp,
    uint8_t * major_version,
    uint8_t * minor_version,
    size_t * header_size
){
  // load from file
  MYFREAD(major_version, sizeof(uint8_t), 1, fp, return 1);
  MYFREAD(minor_version, sizeof(uint8_t), 1, fp, return 1);
  // check version 1.x or 2.x or 3.x
  if(1 != *major_version && 2 != *major_version && 3 != *major_version){
    SNPYIO_ERROR("major version (%hhu) should be 1, 2, or 3\n", *major_version);
    return 1;
  }
  // check version x.0
  if(0 != *minor_version){
    SNPYIO_ERROR("minor version (%hhu) should be 0\n", *minor_version);
    return 1;
  }
  SNPYIO_LOGGING("major version: %hhu\n", *major_version);
  SNPYIO_LOGGING("minor version: %hhu\n", *minor_version);
  *header_size += sizeof(uint8_t) + sizeof(uint8_t);
  return 0;
}

// load and check the header size of the NPY file: HEADER_LEN
static int load_header_len(
    FILE * fp,
    const size_t major_version,
    size_t * header_len,
    size_t * header_size
){
  // buffer size differs based on major_version
  const size_t buf_size =
    1 == major_version
    ? sizeof(uint16_t)
    : sizeof(uint32_t);
  const size_t nitems = buf_size / sizeof(uint8_t);
  // allocate buffer and load corresponding memory size from file
  uint8_t * buf = memory_alloc(nitems, sizeof(uint8_t));
  MYFREAD(buf, sizeof(uint8_t), nitems, fp, return 1);
  // convert endian of loaded buffer when needed
  if(is_big_endian()){
    if(0 != convert_endian(buf_size, buf)){
      SNPYIO_ERROR("convert_endian failed\n");
      return 1;
    }
  }
  // interpret buffer (sequence of uint8_t) as a value having corresponding datatype
  if(1 == major_version){
    // interpret as a 2-byte value
    uint16_t tmp = 0;
    memcpy(&tmp, buf, nitems * sizeof(uint8_t));
    *header_len = (size_t)tmp;
  }else{
    // interpret as a 4-byte value
    uint32_t tmp = 0;
    memcpy(&tmp, buf, nitems * sizeof(uint8_t));
    *header_len = (size_t)tmp;
  }
  memory_free(buf);
  SNPYIO_LOGGING("header_len: %zu\n", *header_len);
  *header_size += buf_size;
  return 0;
}

// load dictionary and padding
static int load_dict_and_padding(
    FILE * fp,
    const size_t header_len,
    uint8_t ** dict_and_padding,
    size_t * header_size
){
  // padding is also loaded to move file pointer forward
  const size_t nitems = header_len / sizeof(uint8_t);
  *dict_and_padding = memory_alloc(nitems, sizeof(uint8_t));
  MYFREAD(*dict_and_padding, sizeof(uint8_t), nitems, fp, return 1);
  *header_size += header_len;
  return 0;
}

// extract dictionary from the mixture of dictionary and padding
static int clip_dict(
    const uint8_t * dict_and_padding,
    const size_t header_len,
    char ** dict_
){
  // check range first
  // e.g., dict_and_padding:
  //   <------------------------ dict ------------------------><- padding ->
  //   {'descr': VALUE, 'fortran_order': VALUE, 'shape': VALUE}           \n
  //   ^                                                      ^
  //   range[0]                                               range[1]
  size_t range[2] = {0, header_len - 1};
  // start, check from the head
  for(range[0] = 0; ; range[0] += 1){
    if(0 == memcmp(
          dict_and_padding + range[0],
          (char []){'{'},
          sizeof(char)
    )){
      break;
    }
    if(header_len - 1 == range[0]){
      SNPYIO_ERROR("%s: '{' not found\n", dict_and_padding);
      return 1;
    }
  }
  // end, check from the tail
  for(range[1] = header_len - 1; ; range[1] -= 1){
    if(0 == memcmp(
          dict_and_padding + range[1],
          (char []){'}'},
          sizeof(char)
    )){
      break;
    }
    if(1 == range[1]){
      // not found or dict is empty
      SNPYIO_ERROR("%s: '}' not found\n", dict_and_padding);
      return 1;
    }
  }
  // copy the dictionary to buffer
  const size_t nitems = range[1] - range[0] + 1;
  *dict_ = memory_alloc(nitems + 1, sizeof(char));
  memcpy(*dict_, dict_and_padding + range[0], nitems);
  (*dict_)[nitems] = null_char;
  return 0;
}

// eliminate "meaningless spaces" (spaces outside quotations)
static int strip_dict(
    const char * dict_,
    char ** dict
){
  // flag dict_ to decide
  //   which part should be / should not be eliminated
  const size_t nitems_ = strlen(dict_);
  bool * is_valid = memory_alloc(nitems_, sizeof(bool));
  for(size_t n = 0; n < nitems_; n++){
    // initially assume all survive
    is_valid[n] = true;
  }
  // e.g., the following spaces are removed here
  //   {'descr': VALUE, 'fortran_order': VALUE, 'shape': VALUE}
  //            ^      ^                ^      ^        ^
  size_t nitems = nitems_;
  bool is_inside[] = {false, false};
  for(size_t n = 0; n < nitems_; n++){
    const char c = dict_[n];
    // inside a pair of single quotations
    if('\'' == c){
      is_inside[0] = !is_inside[0];
      continue;
    }
    // inside a pair of double quotations
    if('"' == c){
      is_inside[1] = !is_inside[1];
      continue;
    }
    // not a space, the information is meaningful
    //   as a member of the dictionary
    if(' ' != c){
      continue;
    }
    // this character is a space but inside key or value
    // these spaces should NOT be removed
    if(is_inside[0] || is_inside[1]){
      continue;
    }
    // this character is a space and outside pair of quotations,
    //   indicating it is meaningless and thus deflagged
    nitems -= 1;
    is_valid[n] = false;
  }
  // copy flagged part from dict_ to dict
  *dict = memory_alloc(nitems + 1, sizeof(char));
  for(size_t n = 0, m = 0; n < nitems_; n++){
    if(is_valid[n]){
      (*dict)[m] = dict_[n];
      m += 1;
    }
  }
  (*dict)[nitems] = null_char;
  memory_free(is_valid);
  return 0;
}

// extract "dict" from "dict_and_padding"
static int polish_dict(
    const uint8_t * dict_and_padding,
    const size_t header_len,
    char ** dict
){
  // extract dictionary part from the raw buffer
  char * dict_ = NULL;
  if(0 != clip_dict(dict_and_padding, header_len, &dict_)){
    return 1;
  }
  // make the dictionary slim for later convenience
  if(0 != strip_dict(dict_, dict)){
    return 1;
  }
  memory_free(dict_);
  SNPYIO_LOGGING("dict: %s\n", *dict);
  return 0;
}

// obtain dictionary value from the given dictionary and key
static int find_dict_value(
    const char * dict,
    const char key[],
    char ** val
){
  // find the starting point of value
  // first find the key
  char * val_s = strstr(dict, key);
  if(NULL == val_s){
    SNPYIO_ERROR("key (%s) not found in dict (%s)\n", key, dict);
    return 1;
  }
  // move pointer to the head of value
  // NOTE: +1 is to account for ":"
  val_s += strlen(key) + 1;
  // find ",", which is the deliminator of python dict,
  //   or "}", which is the terminator of python dict
  // NOTE: "," is used as a deliminator of python tuple
  //   so it is important not to take it as a deliminator
  //   if the parser is inside a tuple
  // initialise end of value as the starting point of it
  char * val_e = val_s;
  bool is_inside[] = {false, false};
  for(int bracket_level = 0; ; val_e += 1){
    const char c = *val_e;
    // check whether we are inside a pair of single quotations
    if('\'' == c){
      is_inside[0] = !is_inside[0];
    }
    // check whether we are inside a pair of double quotations
    if('"' == c){
      is_inside[1] = !is_inside[1];
    }
    // do not check when the tokeniser is inside quotations
    if(is_inside[0] || is_inside[1]){
      continue;
    }
    // otherwise check tuple
    if('(' == c){
      bracket_level += 1;
    }
    if(')' == c){
      bracket_level -= 1;
    }
    if(0 > bracket_level){
      SNPYIO_ERROR("%s: ')' found but corresponding '(' not found\n", dict);
      return 1;
    }
    if(1 < bracket_level){
      SNPYIO_ERROR("%s: nested tuple is not supported\n", dict);
      return 1;
    }
    // we are at the end of val if ',' is found and outside all brackets
    if(',' == c && 0 == bracket_level){
      break;
    }
    // we are at the end of dict if '}' is found
    if('}' == c){
      break;
    }
    if(null_char == c){
      SNPYIO_ERROR("%s: null_char is found before ',' and '}'\n", dict);
      return 1;
    }
  }
  // value ends one character before the terminator
  val_e -= 1;
  // check wrap-around, just in case
  if(val_s + 1 > val_e){
    SNPYIO_ERROR("key %s: no value found\n", key);
    return 1;
  }
  // now we know where val starts and terminates, so extract it
  const size_t nitems = (size_t)(val_e + 1 - val_s) / sizeof(char);
  // +1: null character
  *val = memory_alloc(nitems + 1, sizeof(char));
  memcpy(*val, val_s, nitems * sizeof(char));
  (*val)[nitems] = null_char;
  return 0;
}

// obtain shape from the given dict
static int extract_shape(
    const char * dict,
    size_t * ndim,
    size_t ** shape
){
  char * val = NULL;
  if(0 != find_dict_value(dict, "'shape'", &val)){
    return 1;
  }
  // copy "val" to a buffer "str" after removing parentheses
  const size_t nitems = strlen(val);
  if(2 > nitems){
    SNPYIO_ERROR("number of characters of val: %s, which is a tuple, is less than 2\n", val);
    return 1;
  }
  // with null character (+1), no parentheses (-2): -1
  char * str = memory_alloc(nitems - 1, sizeof(char));
  // +1: skip first "(", copy (nitems - 2) elements
  memcpy(str, val + 1, (nitems - 2) * sizeof(char));
  str[nitems - 2] = null_char;
  memory_free(val);
  // parse "str" to know ndim and shape
  // e.g.,
  //   <empty> -> ndim = 0, shape = NULL
  //   314,    -> ndim = 1, shape[0] = 314
  //   31,4    -> ndim = 2, shape[0] = 31, shape[1] = 4
  //   3,1,4,  -> ndim = 3, shape[0] = 3, shape[1] = 1, shape[2] = 4
  // since ndim is unknown for now, store result as a linked list
  *ndim = 0;
  node_t * shape_ = NULL;
  for(size_t loc = 0; ; ){
    // tokenise
    // set pointer to the current location
    char * buf = str + loc;
    // current location is already the end of string
    if(null_char == *buf){
      break;
    }
    // find the next delimiter and replace it with null character
    for(char * c = buf; null_char != *c; c++, loc++){
      if(',' == *c){
        // replace delimiter with null character
        *c = null_char;
        // next investigation starts one character ahead
        loc++;
        break;
      }
    }
    // try to interpret the NULL-terminated string as a number
    const long long llnum = strtoll(buf, NULL, 10);
    if(0 >= llnum){
      SNPYIO_ERROR("%s: non-positive shape: %lld\n", str, llnum);
      return 1;
    }
    // valid shape, store result to linked list
    const size_t num = (size_t)llnum;
    node_t * new_node = memory_alloc(1, sizeof(node_t));
    new_node->ptr = memory_alloc(1, sizeof(size_t));
    memcpy(new_node->ptr, &num, sizeof(size_t));
    new_node->next = shape_;
    shape_ = new_node;
    (*ndim)++;
  }
  // clean-up buffer
  memory_free(str);
  // convert "shape_" (linked list) to normal pointer "shape"
  if(0 == *ndim){
    *shape = NULL;
  }else{
    *shape = memory_alloc(*ndim, sizeof(size_t));
    for(size_t n = 0; n < *ndim; n++){
      // NOTE: cast from (void *) to (size_t *)
      (*shape)[*ndim - n - 1] = *((size_t *)shape_->ptr);
      node_t *next = shape_->next;
      memory_free(shape_->ptr);
      memory_free(shape_);
      shape_ = next;
    }
  }
  SNPYIO_LOGGING("ndim: %zu\n", *ndim);
  for(size_t n = 0; n < *ndim; n++){
    SNPYIO_LOGGING("shape[%zu]: %zu\n", n, (*shape)[n]);
  }
  return 0;
}

// obtain dtype from the given dict
static int extract_dtype(
    const char * dict,
    char ** dtype
){
  char * val = NULL;
  if(0 != find_dict_value(dict, "'descr'", &val)){
    return 1;
  }
  const size_t nitems = strlen(val);
  *dtype = memory_alloc(nitems + 1, sizeof(char));
  memcpy(*dtype, val, nitems * sizeof(char));
  (*dtype)[nitems] = null_char;
  memory_free(val);
  SNPYIO_LOGGING("dtype: %s\n", *dtype);
  return 0;
}

// obtain is_fortran_order from the given dict
static int extract_is_fortran_order(
    const char * dict,
    bool * is_fortran_order
){
  char * val = NULL;
  if(0 != find_dict_value(dict, "'fortran_order'", &val)){
    return 1;
  }
  const bool  true_is_found = (NULL != strstr(val,  "True"));
  const bool false_is_found = (NULL != strstr(val, "False"));
  if(true_is_found && false_is_found){
    SNPYIO_ERROR("both True and False are found: %s\n", val);
    return 1;
  }
  if((!true_is_found) && (!false_is_found)){
    SNPYIO_ERROR("neither True nor False was found: %s\n", val);
    return 1;
  }
  *is_fortran_order = true_is_found ? true : false;
  memory_free(val);
  SNPYIO_LOGGING("is_fortran_order: %s\n", *is_fortran_order ? "True" : "False");
  return 0;
}

int snpyio_read_header(
    size_t * const ndim,
    size_t ** const shape,
    char ** const dtype,
    bool * const is_fortran_order,
    FILE * const fp,
    size_t * const header_size
){
  if(0 != sanitise_fp(fp)){
    goto err_hndl;
  }
  // NOTE: header_len is the total size of the dictionary and the padding
  //   and thus is different from header_size, which is
  //   the total size of the whole header
  *header_size = 0;
  // load each component of the header
  //   and move file pointer forward
  if(0 != load_magic_string(fp, header_size)){
    goto err_hndl;
  }
  uint8_t major_version = 0;
  uint8_t minor_version = 0;
  if(0 != load_versions(fp, &major_version, &minor_version, header_size)){
    goto err_hndl;
  }
  size_t header_len  = 0;
  if(0 != load_header_len(fp, major_version, &header_len, header_size)){
    goto err_hndl;
  }
  uint8_t * dict_and_padding = NULL;
  if(0 != load_dict_and_padding(fp, header_len, &dict_and_padding, header_size)){
    goto err_hndl;
  }
  // obtain dictionary from the mixture of the dictionary and the padding
  // also non-crutial spaces (spaces outside quotations) are eliminated
  //   e.g., {'descr': '<i4','fortran_order': False,'shape': (3, 5, )}
  //      -> {'descr':'<i4','fortran_order':False,'shape':(3,5,)}
  char * dict = NULL;
  if(0 != polish_dict(dict_and_padding, header_len, &dict)){
    goto err_hndl;
  }
  memory_free(dict_and_padding);
  // extract information which are needed to reconstruct array:
  //   shape, datatype, and memory order of the array
  if(0 != extract_shape(dict, ndim, shape)){
    goto err_hndl;
  }
  if(0 != extract_dtype(dict, dtype)){
    goto err_hndl;
  }
  if(0 != extract_is_fortran_order(dict, is_fortran_order)){
    goto err_hndl;
  }
  // clean-up buffer storing dictionary
  memory_free(dict);
  // detach memories from the memory pool
  // NOTE: user is responsible for deallocating them
  detach_list(*shape);
  detach_list(*dtype);
  return 0;
err_hndl:
  error_handlings();
  return 1;
}

// check positiveness of the array sizes
static int sanitise_shape(
    const size_t ndim,
    const size_t * shape
){
  for(size_t n = 0; n < ndim; n++){
    if(0 >= shape[n]){
      SNPYIO_ERROR("shape[%zu] should be positive\n", n);
      return 1;
    }
  }
  return 0;
}

// check dtype is valid
static int sanitise_dtype(
    const char dtype[]
){
  if(NULL == dtype){
    SNPYIO_ERROR("dtype is NULL\n");
    return 1;
  }
  const size_t nitems = strlen(dtype);
  if(2 > nitems){
    goto err_hndl;
  }
  // should be one of 'xxx' or "xxx"
  if('\'' == dtype[0] && '\'' == dtype[nitems - 1]){
    return 0;
  }
  if('"' == dtype[0] && '"' == dtype[nitems - 1]){
    return 0;
  }
err_hndl:
  SNPYIO_ERROR("dtype (%s) should be a Python string: one of 'xxx' or \"xxx\"\n", dtype);
  return 1;
}

// create a value of a dictionary key: "descr"
static int create_descr_value(
    char ** value,
    const char dtype[]
){
  const size_t nitems = strlen(dtype);
  // +1: null character
  *value = memory_alloc(nitems + 1, sizeof(char));
  memcpy(*value, dtype, nitems * sizeof(char));
  (*value)[nitems] = null_char;
  return 0;
}

// create a value of a dictionary key: "fortran_order"
static int create_fortran_order_value(
    char ** value,
    const bool is_fortran_order
){
  // True or False
  const char * string = is_fortran_order ? "True" : "False";
  const size_t nitems = strlen(string);
  // +1: null character
  *value = memory_alloc(nitems + 1, sizeof(char));
  memcpy(*value, string, nitems * sizeof(char));
  (*value)[nitems] = null_char;
  return 0;
}

// create a value of a dictionary key: "shape"
static int create_shape_value(
    char ** value,
    const size_t ndim,
    const size_t * shape
){
  // e.g.:
  //   0D array: ndim = 0, *dims = NULL  -> ()
  //   1D array: ndim = 1, *dims = {5}   -> (5,)
  //   2D array: ndim = 2, *dims = {5,2} -> (5,2,)
  //   from left to right,
  //     from outer (less contiguous) to inner (contiguous)
  // count number of digits (e.g., 5: 1 digit, 15: 2 digits)
  //   of shape in each direction
  size_t * n_digits = memory_alloc(ndim, sizeof(size_t));
  for(size_t n = 0; n < ndim; n++){
    // simple way to compute digits,
    //   dividing by 10 as many times as possible
    size_t num = shape[n];
    n_digits[n] = 1;
    while(num /= 10){
      n_digits[n] += 1;
    }
  }
  // compute total number of characters
  //   i.e., memory size to be allocated
  size_t nitems = 2; // at least "(", ")"
  for(size_t n = 0; n < ndim; n++){
    // number of digits in n-th direction
    // with comma (+1)
    nitems += n_digits[n] + 1;
  }
  // allocate memory and assign values
  *value = memory_alloc(nitems + 1, sizeof(char));
  for(size_t n = 0, offset = 1; n < ndim; n++){
    // assign size of the array in each direction to "value"
    //   after converting the integer to characters, e.g., 128 -> "128"
    const size_t n_digits_ = n_digits[n];
    // + "," and null character
    char * buf = memory_alloc(n_digits_ + 2, sizeof(char));
    // including ","
    if((int)(n_digits_ + 1) != snprintf(buf, n_digits_ + 2, "%zu,", shape[n])){
      SNPYIO_ERROR("snprintf failed\n");
      return 1;
    }
    // copy result excluding null character
    memcpy((*value) + offset, buf, (n_digits_ + 1) * sizeof(char));
    offset += n_digits_ + 1;
    memory_free(buf);
  }
  // clean-up buffer
  memory_free(n_digits);
  // first character is a parenthesis
  (*value)[         0] = '(';
  // last-1 character is a parenthesis
  (*value)[nitems - 1] = ')';
  // last character is null character
  (*value)[    nitems] = null_char;
  return 0;
}

// prepare dictionary which will be embedded in the header
static int create_dict(
    char ** dict,
    size_t * n_dict,
    const size_t ndim,
    const size_t * shape,
    const char dtype[],
    const bool is_fortran_order
){
  // "dict" contains information which is necessary to recover the original array,
  //   1. datatype, 2. memory ordering, and 3. shape of the data
  // It is a python-like dictionary, whose structure is a pair of key and value:
  // --- -------------- -----------------------------------------------------------------
  //  1.  descr         Datatype, e.g., '<f8', 'float64'
  //  2.  fortran_order Memory order, True or False (usually False)
  //  3.  shape         A tuple having number of elements in each direction, e.g., (5,2,)
  // See corresponding function for how they are created
  // Also the number of elements of the dict is returned (to be consistent with "create_padding")
  // buffers to store values
  char * descr_value         = NULL;
  char * fortran_order_value = NULL;
  char * shape_value         = NULL;
  // create dictionary values,
  //   in which inputs are evaluated and sanitised
  // create value of data type
  if(0 != create_descr_value(&descr_value, dtype)){
    SNPYIO_ERROR("create_descr_value failed\n");
    return 1;
  }
  // create value of memory order
  if(0 != create_fortran_order_value(&fortran_order_value, is_fortran_order)){
    SNPYIO_ERROR("create_fortran_order_value failed\n");
    return 1;
  }
  // create value of data sizes
  if(0 != create_shape_value(&shape_value, ndim, shape)){
    SNPYIO_ERROR("create_shape_value failed\n");
    return 1;
  }
  // assign all elements (strings) which compose dict
  const char * elements[] = {
    "{",
    "'descr'",         ":", descr_value,         ",",
    "'fortran_order'", ":", fortran_order_value, ",",
    "'shape'",         ":", shape_value,         ",",
    "}",
  };
  const size_t nitems = sizeof(elements) / sizeof(elements[0]);
  // check total number of characters of
  //   {'descr':VALUE,'fortran_order':VALUE,'shape':VALUE}
  //   to allocate dict
  // NOTE: n_chars_dict is the number of characters of dict
  //   INCLUDING the last null character, while n_dict = strlen(dict),
  //   EXCLUDING the last null character.
  //   Thus n_dict = n_chars_dict - 1
  size_t n_chars_dict = 0;
  for(size_t n = 0; n < nitems; n++){
    n_chars_dict += strlen(elements[n]);
  }
  // allocate dict and assign above "elements"
  // +1: null character
  *dict = memory_alloc(n_chars_dict + 1, sizeof(char));
  for(size_t n = 0, offset = 0; n < nitems; n++){
    const size_t n_chars = strlen(elements[n]);
    memcpy((*dict) + offset, elements[n], n_chars * sizeof(char));
    offset += n_chars;
  }
  (*dict)[n_chars_dict] = null_char;
  // clean-up all working memories
  memory_free(descr_value);
  memory_free(fortran_order_value);
  memory_free(shape_value);
  // as the length of "dict", use length WITHOUT null character
  *n_dict = strlen(*dict);
  SNPYIO_LOGGING("dict: %s\n", *dict);
  SNPYIO_LOGGING("size: %zu\n", *n_dict);
  return 0;
}

// prepare padding after dictionary and decide major version
static int create_padding(
    uint8_t ** padding,
    size_t * n_padding,
    uint8_t * major_version,
    size_t * header_size,
    const size_t n_dict
){
  // The following relation holds for the header size
  //   header_size =
  //     + sizeof(magic string)        (= 6           ) bytes
  //     + sizeof(major_version)       (= 1           ) byte
  //     + sizeof(minor_version)       (= 1           ) byte
  //     + sizeof(header_len)          (= 2 or 4      ) bytes
  //     + strlen(dict) * sizeof(char) (= strlen(dict)) bytes
  //     + n_padding * sizeof(uint8_t) (= n_padding   ) bytes
  //   is divisible by header_block_size
  // to be so I need some paddings consisted of
  //   some spaces ' ' and one newline '\n',
  //   whose length (number of elements) is returned
  // size of each element is computed
  const size_t n_magic_string = strlen(magic_string);
  const size_t size_magic_string  = n_magic_string * sizeof(   char);
  const size_t size_major_version = 1              * sizeof(uint8_t);
  const size_t size_minor_version = 1              * sizeof(uint8_t);
  const size_t size_dict          = n_dict         * sizeof(   char);
  // reject too large dict
  if(UINT_MAX - header_block_size < size_dict){
    SNPYIO_ERROR("size of dictionary is huge (%zu)\n", size_dict);
    return 1;
  }
  // decide major version and datatype of HEADER_LEN
  // large portion of the header is occupied by dict
  // so check dict size, and if it is larger than USHRT_MAX - header_block_size,
  //   use major_version = 2
  size_t size_header_len = 0;
  if(USHRT_MAX - header_block_size < size_dict){
    *major_version = 2;
    size_header_len = sizeof(uint32_t);
  }else{
    *major_version = 1;
    size_header_len = sizeof(uint16_t);
  }
  // compute size of all data except padding
  const size_t size_except_padding =
    + size_magic_string
    + size_major_version
    + size_minor_version
    + size_header_len
    + size_dict;
  // decide total size of the header, which should be header_block_size x N
  // increase total size by header_block_size until becoming larger than size_except_padding
  // NOTE: size_padding == 0 is NOT allowed since '\n' is necessary at the end
  //   thus the condition to continue loop is "<=", not "<"
  *header_size = 0;
  while(*header_size <= size_except_padding){
    *header_size += header_block_size;
  }
  const size_t size_padding = *header_size - size_except_padding;
  // create padding
  *n_padding = size_padding / sizeof(uint8_t);
  *padding = memory_alloc(*n_padding, sizeof(uint8_t));
  // many ' 's
  memset(*padding, ' ', (*n_padding - 1) * sizeof(uint8_t));
  // last '\n'
  (*padding)[*n_padding - 1] = '\n';
  SNPYIO_LOGGING("padding, size: %zu\n", *n_padding);
  return 0;
}

// HEADER_LEN = n_dict + n_padding, which is written in little-endian
static int create_header_len(
    uint8_t ** header_len,
    size_t * n_header_len,
    const uint8_t major_version,
    const size_t n_dict,
    const size_t n_padding
){
  // reject too large dict / padding sizes
  // Here "too large" means header size (not data size)
  //   is larger than approx. 2GB, which would not happen normally
  if(UINT_MAX / 2 <= n_dict){
    SNPYIO_ERROR("dictionary size is huge (%zu)\n", n_dict);
    return 1;
  }
  // padding is to make header size header_block_size x N
  // so it should not exceed header_block_size
  if(header_block_size < n_padding){
    SNPYIO_ERROR("padding size is huge (%zu)\n", n_padding);
    return 1;
  }
  // compute header_len and store as an array of uint8_t
  if(1 == major_version){
    // major version 1, use uint16_t to store header_len
    const uint16_t header_len_ = (uint16_t)(n_dict + n_padding);
    *n_header_len = sizeof(uint16_t) / sizeof(uint8_t);
    *header_len = memory_alloc(*n_header_len, sizeof(uint8_t));
    memcpy(*header_len, &header_len_, *n_header_len);
    SNPYIO_LOGGING("header_len (uint16_t): %hu\n", header_len_);
  }else{
    // major version 2, use uint32_t to store header_len
    const uint32_t header_len_ = (uint32_t)(n_dict + n_padding);
    *n_header_len = sizeof(uint32_t) / sizeof(uint8_t);
    *header_len = memory_alloc(*n_header_len, sizeof(uint8_t));
    memcpy(*header_len, &header_len_, *n_header_len);
    SNPYIO_LOGGING("header_len (uint32_t): %u\n", header_len_);
  }
  // convert endian of buffer which will be written if needed
  if(is_big_endian()){
    if(0 != convert_endian(sizeof(*header_len), header_len)){
      SNPYIO_ERROR("convert_endian failed\n");
      return 1;
    }
  }
  return 0;
}

int snpyio_write_header(
    const size_t ndim,
    const size_t * const shape,
    const char dtype[],
    const bool is_fortran_order,
    FILE * const fp,
    size_t * const header_size
){
  // check parameters given by user
  if(0 != sanitise_shape(ndim, shape)){
    goto err_hndl;
  }
  if(0 != sanitise_dtype(dtype)){
    goto err_hndl;
  }
  if(0 != sanitise_fp(fp)){
    goto err_hndl;
  }
  // magic string
  const size_t n_magic_string = strlen(magic_string);
  // minor_version, always 0
  const uint8_t minor_version = 0;
  // dictionary (and its size)
  char * dict = NULL;
  size_t n_dict = 0;
  if(0 != create_dict(&dict, &n_dict, ndim, shape, dtype, is_fortran_order)){
    goto err_hndl;
  }
  // padding (and its size)
  // NOTE: major version is also determined here
  uint8_t * padding = NULL;
  size_t n_padding = 0;
  uint8_t major_version = 0;
  *header_size = 0;
  if(0 != create_padding(&padding, &n_padding, &major_version, header_size, n_dict)){
    goto err_hndl;
  }
  // comptue header_len, sum of n_dict + n_padding
  // NOTE: data type can vary depending on the major version,
  //   and as a result a series of uint8_t is used to store
  uint8_t * header_len = NULL;
  size_t n_header_len = 0;
  if(0 != create_header_len(&header_len, &n_header_len, major_version, n_dict, n_padding)){
    goto err_hndl;
  }
  // store all information to a continuous buffer
  //   and write it to the given file stream
  //   to reduce the number of write calls
  uint8_t * buffer = memory_alloc(1, *header_size);
  const void * elements[] = {
    magic_string,
    &major_version,
    &minor_version,
    header_len,
    dict,
    padding,
  };
  const size_t sizes[] = {
    n_magic_string * sizeof(   char),
    1              * sizeof(uint8_t),
    1              * sizeof(uint8_t),
    n_header_len   * sizeof(uint8_t),
    n_dict         * sizeof(   char),
    n_padding      * sizeof(uint8_t),
  };
  for(size_t n = 0, offset = 0; n < sizeof(elements) / sizeof(elements[0]); n++){
    const size_t size = sizes[n];
    memcpy(buffer + offset, elements[n], size);
    offset += size;
  }
  if(1 != fwrite(buffer, *header_size, 1, fp)){
    SNPYIO_ERROR("failed to write header to file\n");
    goto err_hndl;
  }
  memory_free(buffer);
  // clean-up all buffers
  memory_free(dict);
  memory_free(padding);
  memory_free(header_len);
  SNPYIO_LOGGING("header_size: %zu\n", *header_size);
  return 0;
err_hndl:
  error_handlings();
  return 1;
}

int snpyio_skip_header(
    FILE * const fp
) {
  size_t ndim = 0;
  size_t * shape = NULL;
  char * dtype = NULL;
  bool is_fortran_order = false;
  size_t header_size = 0;
  if (0 != snpyio_read_header(&ndim, &shape, &dtype, &is_fortran_order, fp, &header_size)) {
    goto err_hndl;
  }
  memory_free(shape);
  memory_free(dtype);
  return 0;
err_hndl:
  error_handlings();
  return 1;
}

int snpyio_r_header(
    size_t * const ndim,
    size_t ** const shape,
    char ** const dtype,
    bool * const is_fortran_order,
    FILE * const fp,
    size_t * const header_size
) {
  return snpyio_read_header(ndim, shape, dtype, is_fortran_order, fp, header_size);
}

int snpyio_w_header(
    const size_t ndim,
    const size_t * const shape,
    const char dtype[],
    const bool is_fortran_order,
    FILE * const fp,
    size_t * const header_size
) {
  return snpyio_write_header(ndim, shape, dtype, is_fortran_order, fp, header_size);
}

