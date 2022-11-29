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

__device__ bool filename_equal(FileSystem* fs, char* s, char* t);
__device__ int filename_cmp(FileSystem* fs, char* s, char* t);

class SUPERBLOCK {
   public:
    static __device__ u32 get_storage_end_fp(FileSystem* fs);
    static __device__ u32 malloc_free_DB(FileSystem* fs);
    static __device__ void malloc_DB(FileSystem* fs, u32 blockNum);
    static __device__ void free_DB(FileSystem* fs, u32 blockNum);

   private:
    static __device__ bool isMalloced(FileSystem* fs, u32 blockNum);
};

class DB {
   public:
    static __device__ void write_byte(FileSystem* fs,
                                      uchar s,
                                      u32 blockNum,
                                      u32 pos);
    static __device__ u32 get_blockNum(FileSystem* fs, u32 fp);
    static __device__ void move_DB(FileSystem* fs, u32 desblock, u32 tarblock);
    static __device__ u32 get_fp(FileSystem* fs, u32 db);

   private:
    ;
};

class FCB {
   public:
    static __device__ u32 new_FCB(FileSystem* fs, char* filename);
    static __device__ void get_filename(FileSystem* fs,
                                        u32 FCBNum,
                                        char* filename);
    static __device__ u32 get_db(FileSystem* fs, u32 FCBNum);
    static __device__ u32 get_size(FileSystem* fs, u32 FCBNum);
    static __device__ void set_filename(FileSystem* fs,
                                        u32 FCBNum,
                                        char* filename);
    static __device__ void set_db(FileSystem* fs, u32 FCBNum, u32 db);
    static __device__ void set_size(FileSystem* fs, u32 FCBNum, u32 size);

    static __device__ u32 search_by_filename(FileSystem* fs, char* filename);
    static __device__ u32 search_by_db(FileSystem* fs, u32 db);
    static __device__ void compact(FileSystem* fs, u32 FCBNum);
    static __device__ u32 get_nFCB(FileSystem* fs);

   private:
    static __device__ void move_FCB(FileSystem* fs, u32 desFCB, u32 tarFCB);
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