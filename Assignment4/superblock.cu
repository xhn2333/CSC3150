#include "file_system.h"

__device__ u32 SUPERBLOCK::get_storage_end_fp(FileSystem* fs) {
    for (int i = 0; i < fs->SUPERBLOCK_SIZE * 8; ++i) {
        if (!isMalloced(fs, i))
            return (i + 1) * fs->STORAGE_BLOCK_SIZE + fs->FILE_BASE_ADDRESS;
    }
    return -1;
}

__device__ void SUPERBLOCK::malloc_DB(FileSystem* fs, u32 blockNum) {
    fs->volume[blockNum / 8] =
        fs->volume[blockNum / 8] | (1 << (7 - blockNum % 8));
    return;
}

__device__ bool SUPERBLOCK::isMalloced(FileSystem* fs, u32 blockNum) {
    return fs->volume[blockNum / 8] & (1 << (7 - blockNum % 8));
}

__device__ void SUPERBLOCK::free_DB(FileSystem* fs, u32 blockNum) {
    fs->volume[blockNum / 8] =
        (fs->volume[blockNum / 8] | (1 << (7 - blockNum % 8))) -
        (1 << (7 - blockNum % 8));
    return;
}

__device__ u32 SUPERBLOCK::malloc_free_DB(FileSystem* fs) {
    for (int i = 0; i < fs->SUPERBLOCK_SIZE * 8; ++i) {
        if (!isMalloced(fs, i)) {
            SUPERBLOCK::malloc_DB(fs, i);
            return i;
        }
    }
    return -1;
}