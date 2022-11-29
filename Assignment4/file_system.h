#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2

struct FileSystem {
    uchar* volume;
    int SUPERBLOCK_SIZE;
    int FCB_SIZE;
    int FCB_ENTRIES;
    int STORAGE_SIZE;
    int STORAGE_BLOCK_SIZE;
    int MAX_FILENAME_SIZE;
    int MAX_FILE_NUM;
    int MAX_FILE_SIZE;
    int FILE_BASE_ADDRESS;
};

__device__ void removeCompact(FileSystem* fs, u32 FCB_pos);

class SUPERBLOCK {
   public:
    static __device__ u32 get_storage_end_fp(FileSystem* fs);
    static __device__ void update_storage_end_fp(FileSystem* fs, u32 fp);
};

class FCB {
   public:
    static __device__ u32 create_FCB(FileSystem* fs, char* filename, u32 size);
    static __device__ u32 get_size(FileSystem* fs, u32 FCB_pos);
    static __device__ u32 get_fp(FileSystem* fs, u32 FCB_pos);
    static __device__ void update_fp(FileSystem* fs, u32 FCB_pos, u32 fp);
    static __device__ int search_file(FileSystem* fs, char* filename);
    static __device__ int search_file_by_fp(FileSystem* fs, u32 fp);
    static __device__ void update_size(FileSystem* fs, u32 FCB_pos, u32 size);
    static __device__ void get_filename(FileSystem* fs, u32 FCB_pos, char* buf);

   private:
    static __device__ u32 get_empty_FCB(FileSystem* fs);
    static __device__ u32 write_filename(FileSystem* fs,
                                         u32 FCB_pos,
                                         char* filename);

    static __device__ u32 malloc_storage_block(FileSystem* fs);
};

__device__ void fs_init(FileSystem* fs,
                        uchar* volume,
                        int SUPERBLOCK_SIZE,
                        int FCB_SIZE,
                        int FCB_ENTRIES,
                        int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE,
                        int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM,
                        int MAX_FILE_SIZE,
                        int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem* fs, char* s, int op);
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem* fs, int op);
__device__ void fs_gsys(FileSystem* fs, int op, char* s);

#endif