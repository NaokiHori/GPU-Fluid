#include <errno.h> // errno, EEXIST
#include <stdbool.h> // false
#include <stdio.h> // snprintf
#include <string.h> // strlen
#include <sys/stat.h> // mode_t, S_IRWXU, S_IRWXG, S_IRWXO
#include "domain.h"
#include "memory.h"
#include "./save.h"
#include "./save/snpyio.h"

#define ROOT_DIRECTORY "output/save/"

static int concat_dir_name(
    const size_t directory_index,
    char ** const dir_name
) {
  const char prefix[] = ROOT_DIRECTORY;
  const int ndigits = 10;
  const int nchars = strlen(prefix) + ndigits + 1;
  *dir_name = memory_alloc(nchars, sizeof(char));
  (*dir_name)[nchars - 1] = '\0';
  if (nchars - 1 != snprintf(*dir_name, nchars, "%s%0*zu", prefix, ndigits, directory_index)) {
    return 1;
  }
  return 0;
}

static int create_directory(
    const char dir_name[]
) {
  errno = 0;
  if (0 != mkdir(dir_name, S_IRWXU | S_IRWXG | S_IRWXO)) {
    perror(dir_name);
    // treat EEXIST as expected to override present files
    if (EEXIST != errno) {
      return 1;
    }
  }
  return 0;
}

static int write_npy_file(
    const char dir_name[],
    const char dset_name[],
    const size_t ndims,
    const size_t * shape,
    const char dtype[],
    const size_t size,
    const void * data
){
  int error_code = 0;
  char * file_name = NULL;
  FILE * fp = NULL;
  size_t header_size = 0;
  // assign file_name
  {
    const char slash[] = {"/"};
    const char suffix[] = {".npy"};
    const int nchars =
      + strlen( dir_name)
      + strlen(    slash)
      + strlen(dset_name)
      + strlen(   suffix)
      + 1;
    file_name = memory_alloc(nchars, sizeof(char));
    file_name[nchars - 1] = '\0';
    if (nchars - 1 != snprintf(file_name, nchars, "%s%s%s%s", dir_name, slash, dset_name, suffix)) {
      error_code = 1;
      goto abort;
    }
  }
  // write npy header
  {
    errno = 0;
    fp = fopen(file_name, "w");
    if (NULL == fp) {
      perror(file_name);
      error_code = 1;
      goto abort;
    }
    size_t header_size = 0;
    if (0 != snpyio_w_header(ndims, shape, dtype, false, fp, &header_size)) {
      error_code = 1;
      goto abort;
    }
    fclose(fp);
  }
  // append main data
  {
    errno = 0;
    fp = fopen(file_name, "a");
    if (NULL == fp) {
      perror(file_name);
      error_code = 1;
      goto abort;
    }
    if (0 != fseek(fp, (long)header_size, SEEK_SET)) {
      error_code = 1;
      goto abort;
    }
    size_t nitems = 1;
    for (size_t dim = 0; dim < ndims; dim++) {
      nitems *= shape[dim];
    }
    if (nitems != fwrite(data, size, nitems, fp)) {
      error_code = 1;
      goto abort;
    }
  }
abort:
  memory_free(file_name);
  if (NULL != fp) {
    fclose(fp);
  }
  return error_code;
}

int save(
    const size_t directory_index,
    const size_t step,
    const double time,
    const domain_t * const domain,
    const flow_field_t * const flow_field
) {
  int error_code = 0;
  char * dir_name = NULL;
  if (0 != concat_dir_name(directory_index, &dir_name)) {
    error_code = 1;
    goto abort;
  }
  if (0 != create_directory(dir_name)) {
    error_code = 1;
    goto abort;
  }
  const size_t nx = domain->x_coordinate.nitems;
  const size_t ny = domain->y_coordinate.nitems;
  const size_t nz = domain->z_coordinate.nitems;
  write_npy_file(dir_name, "step", 0, NULL, "'<u8'", sizeof(size_t), &step);
  write_npy_file(dir_name, "time", 0, NULL, "'<f8'", sizeof(double), &time);
  write_npy_file(dir_name, "ux", NDIMS, (size_t [NDIMS]){nz + 2, ny + 2, nx + 2}, "'<f4'", sizeof(float), [flow_field->ux contents]);
  write_npy_file(dir_name, "uy", NDIMS, (size_t [NDIMS]){nz + 2, ny + 2, nx + 2}, "'<f4'", sizeof(float), [flow_field->uy contents]);
  write_npy_file(dir_name, "uz", NDIMS, (size_t [NDIMS]){nz + 2, ny + 2, nx + 2}, "'<f4'", sizeof(float), [flow_field->uz contents]);
  write_npy_file(dir_name, "p", NDIMS, (size_t [NDIMS]){nz + 2, ny + 2, nx + 2}, "'<f4'", sizeof(float), [flow_field->p contents]);
  write_npy_file(dir_name, "temperature", NDIMS, (size_t [NDIMS]){nz + 2, ny + 2, nx + 2}, "'<f4'", sizeof(float), [flow_field->temperature contents]);
  if (0 == directory_index) {
    write_npy_file(ROOT_DIRECTORY, "x_centers", 1, (size_t [NDIMS]){nx + 2}, "'<f8'", sizeof(double), domain->x_coordinate.centers);
    write_npy_file(ROOT_DIRECTORY, "x_faces", 1, (size_t [NDIMS]){nx + 2}, "'<f8'", sizeof(double), domain->x_coordinate.faces);
    write_npy_file(ROOT_DIRECTORY, "y_centers", 1, (size_t [NDIMS]){ny + 2}, "'<f8'", sizeof(double), domain->y_coordinate.centers);
    write_npy_file(ROOT_DIRECTORY, "y_faces", 1, (size_t [NDIMS]){ny + 2}, "'<f8'", sizeof(double), domain->y_coordinate.faces);
    write_npy_file(ROOT_DIRECTORY, "z_centers", 1, (size_t [NDIMS]){nz + 2}, "'<f8'", sizeof(double), domain->z_coordinate.centers);
    write_npy_file(ROOT_DIRECTORY, "z_faces", 1, (size_t [NDIMS]){nz + 2}, "'<f8'", sizeof(double), domain->z_coordinate.faces);
  }
abort:
  memory_free(dir_name);
  return error_code;
}

