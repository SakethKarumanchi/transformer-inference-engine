/* safetensors.h -- weight-file reader, Stage 1.
 *
 * Reads a safetensors weight file: the header length prefix, the JSON header,
 * the tensor descriptor table, and the bytes of any tensor. Nothing in this
 * header knows what model it is reading -- no tensor name, no dimension and no
 * architecture constant appears here. Everything specific to a model comes out
 * of the file at run time.
 *
 * Every structural fact this reader relies on was read out of the file on disk
 * (models/gpt2/model.safetensors, 548105171 bytes) rather than taken from prior
 * knowledge of the format. What the file shows:
 *
 *   bytes 0..7        header length N, unsigned 64-bit LITTLE ENDIAN.
 *                     Observed N = 14283.
 *   bytes 8..8+N-1    the header, UTF-8 JSON, one object. Observed: pure ASCII,
 *                     no whitespace between tokens, no backslash escape of any
 *                     kind, 161 members -- 160 tensors plus one "__metadata__"
 *                     member whose value is an object of string values
 *                     ({"format":"pt"}).
 *   byte 8+N onward   the data segment, to end of file.
 *
 * Each tensor member is an object with exactly three keys:
 *     "dtype"         a string. Observed: "F32" for all 160 tensors.
 *     "shape"         an array of non-negative integers, length 1 to 4 observed.
 *     "data_offsets"  an array of exactly two non-negative integers, [begin,end),
 *                     RELATIVE TO THE START OF THE DATA SEGMENT, not to the file.
 *                     Observed begin minimum 0 and end maximum 548090880, and
 *                     8 + 14283 + 548090880 == the file size exactly.
 *
 * Alignment: the file exhibits NONE. 8 + N = 14291, which is not a multiple of
 * 8, 16 or 64, so the data segment does not begin on an aligned boundary and no
 * padding is present. A caller that needs aligned floats must copy the bytes
 * out; st_tensor_read does exactly that.
 */
#ifndef TIE_SAFETENSORS_H
#define TIE_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bounds, generous relative to the file this stage reads. They bound the
 * parser's own tables; they are not model parameters. */
#define ST_MAX_NAME  256
#define ST_MAX_DIMS  8

typedef enum {
    ST_OK = 0,
    ST_ERR_OPEN,              /* the file could not be opened                 */
    ST_ERR_READ,              /* a read returned short or failed              */
    ST_ERR_TRUNCATED_HEADER,  /* 8 + header length exceeds the file size      */
    ST_ERR_BAD_JSON,          /* the header is not the JSON subset below      */
    ST_ERR_BAD_DTYPE,         /* a dtype string this reader does not know     */
    ST_ERR_BAD_SHAPE,         /* missing/oversized shape, or negative extent  */
    ST_ERR_RANGE,             /* a declared byte range leaves the data segment */
    ST_ERR_SIZE_MISMATCH,     /* declared length != product(shape) * dtype size */
    ST_ERR_NOT_FOUND,         /* lookup by name found nothing                 */
    ST_ERR_TOO_MANY_DIMS,     /* shape longer than ST_MAX_DIMS                */
    ST_ERR_NAME_TOO_LONG,     /* tensor name longer than ST_MAX_NAME - 1      */
    ST_ERR_NOMEM,             /* allocation failed                            */
    ST_ERR_ARG                /* null or otherwise unusable argument          */
} st_status;

/* The dtype spellings the header may carry. Read from the file as a string and
 * mapped here; ST_DTYPE_UNKNOWN is never stored in a descriptor, an unknown
 * spelling fails the parse with ST_ERR_BAD_DTYPE instead. */
typedef enum {
    ST_DTYPE_UNKNOWN = 0,
    ST_DTYPE_BOOL, ST_DTYPE_U8, ST_DTYPE_I8,
    ST_DTYPE_U16, ST_DTYPE_I16, ST_DTYPE_F16, ST_DTYPE_BF16,
    ST_DTYPE_U32, ST_DTYPE_I32, ST_DTYPE_F32,
    ST_DTYPE_U64, ST_DTYPE_I64, ST_DTYPE_F64
} st_dtype;

typedef struct {
    char     name[ST_MAX_NAME];
    st_dtype dtype;
    int      n_dims;
    int64_t  dims[ST_MAX_DIMS];
    uint64_t begin;        /* declared data_offsets[0], relative to the data segment */
    uint64_t end;          /* declared data_offsets[1], relative to the data segment */
    uint64_t file_offset;  /* begin + the data segment's own file offset            */
    uint64_t nbytes;       /* end - begin                                           */
} st_tensor;

typedef struct st_file st_file;

/* Opens the file, parses the header, validates every declared range, and builds
 * the descriptor table. On failure *out is left NULL and nothing is leaked. */
st_status st_open(const char *path, st_file **out);
void      st_close(st_file *f);

size_t           st_count(const st_file *f);
const st_tensor *st_at(const st_file *f, size_t index);
st_status        st_find(const st_file *f, const char *name, const st_tensor **out);

/* Copies a tensor's bytes into dst. dst_bytes must be at least t->nbytes.
 * The copy is what makes the unaligned data segment safe to read as floats. */
st_status st_tensor_read(const st_file *f, const st_tensor *t,
                         void *dst, size_t dst_bytes);

/* File-level facts, as read rather than as assumed. */
uint64_t st_header_length(const st_file *f);   /* the 64-bit prefix's value      */
uint64_t st_data_offset(const st_file *f);     /* 8 + header length              */
uint64_t st_file_size(const st_file *f);       /* size on disk, in bytes         */

/* The "__metadata__" member, if the file carries one: value for a key, or NULL.
 * It is excluded from the tensor count and from enumeration. */
const char *st_metadata(const st_file *f, const char *key);
int         st_has_metadata(const st_file *f);

size_t      st_dtype_size(st_dtype d);
const char *st_dtype_name(st_dtype d);
const char *st_strerror(st_status s);

#ifdef __cplusplus
}
#endif

#endif /* TIE_SAFETENSORS_H */
