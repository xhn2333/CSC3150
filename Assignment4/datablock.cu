#include "file_system.h"

__device__ void DB::write_byte(FileSystem* fs, uchar s, u32 blockNum, u32 pos) {
    fs->volume[fs->FILE_BASE_ADDRESS + blockNum * (fs->STORAGE_BLOCK_SIZE) +
               pos] = s;
    return;
}

__device__ u32 DB::get_blockNum(FileSystem* fs, u32 fp) {
    u32 blockNum = (fp - fs->FILE_BASE_ADDRESS) / fs->STORAGE_BLOCK_SIZE;
    return blockNum;
}

__device__ u32 DB::get_fp(FileSystem* fs, u32 db) {
    u32 fp = fs->FILE_BASE_ADDRESS + db * fs->STORAGE_BLOCK_SIZE;
    return fp;
}

__device__ void DB::move_DB(FileSystem* fs, u32 desblock, u32 tarblock) {
    for (int i = 0; i < fs->STORAGE_BLOCK_SIZE; ++i) {
        u32 des = desblock * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS + i;
        u32 tar = tarblock * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS + i;
        fs->volume[des] = fs->volume[tar];
        fs->volume[tar] = '\0';
    }
    return;
}