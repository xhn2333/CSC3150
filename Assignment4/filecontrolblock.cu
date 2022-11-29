#include "file_system.h"

/*
    FCB:
        0-19:   filename
        20-23:  size
        24-27:  db
*/

/*
    SUPERBLOCK_SIZE 4096  // 32K/8 bits = 4 K
    FCB_SIZE 32           // 32 bytes per FCB
    FCB_ENTRIES 1024
    VOLUME_SIZE 1085440  // 4096+32768+1048576
    STORAGE_BLOCK_SIZE 32

    MAX_FILENAME_SIZE 20
    MAX_FILE_NUM 1024
    MAX_FILE_SIZE 1048576

    FILE_BASE_ADDRESS 36864  // 4096+32768

*/

__device__ u32 FCB::new_FCB(FileSystem* fs, char* filename) {
    u32 FCBNum = -1;
    if (FCB::get_nFCB(fs) < fs->MAX_FILE_NUM) {
        FCBNum = FCB::get_nFCB(fs);
        FCB::set_filename(fs, FCBNum, filename);
        FCB::set_size(fs, FCBNum, 0);
        u32 db = SUPERBLOCK::malloc_free_DB(fs);
        FCB::set_db(fs, FCBNum, db);
    }
    return FCBNum;
}

__device__ void FCB::get_filename(FileSystem* fs, u32 FCBNum, char* filename) {
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    for (int i = 0;
         i < fs->MAX_FILENAME_SIZE && fs->volume[i + FCB_pos] != '\0'; ++i) {
        filename[i] = fs->volume[i + FCB_pos];
    }
    return;
}
__device__ void FCB::set_filename(FileSystem* fs, u32 FCBNum, char* filename) {
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    for (int i = 0; i < fs->MAX_FILENAME_SIZE && filename[i] != '\0'; ++i) {
        fs->volume[i + FCB_pos] = filename[i];
    }
    return;
}

__device__ u32 FCB::get_db(FileSystem* fs, u32 FCBNum) {
    u32 db = fs->FILE_BASE_ADDRESS;
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    db = fs->volume[FCB_pos + 24] * (1 << 24) +
         fs->volume[FCB_pos + 25] * (1 << 16) +
         fs->volume[FCB_pos + 26] * (1 << 8) +
         fs->volume[FCB_pos + 27] * (1 << 0);
    return db;
}

__device__ void FCB::set_db(FileSystem* fs, u32 FCBNum, u32 db) {
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    fs->volume[FCB_pos + 27] = db % (1 << 8);
    db >>= 8;
    fs->volume[FCB_pos + 26] = db % (1 << 8);
    db >>= 8;
    fs->volume[FCB_pos + 25] = db % (1 << 8);
    db >>= 8;
    fs->volume[FCB_pos + 24] = db % (1 << 8);
    return;
}

__device__ u32 FCB::get_size(FileSystem* fs, u32 FCBNum) {
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    u32 size = fs->volume[FCB_pos + 20] * (1 << 24) +
               fs->volume[FCB_pos + 21] * (1 << 16) +
               fs->volume[FCB_pos + 22] * (1 << 8) +
               fs->volume[FCB_pos + 23] * (1 << 0);
    return size;
}

__device__ void FCB::set_size(FileSystem* fs, u32 FCBNum, u32 size) {
    u32 FCB_pos = FCBNum * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
    fs->volume[FCB_pos + 23] = size % (1 << 8);
    size >>= 8;
    fs->volume[FCB_pos + 22] = size % (1 << 8);
    size >>= 8;
    fs->volume[FCB_pos + 20] = size % (1 << 8);
    size >>= 8;
    fs->volume[FCB_pos + 21] = size % (1 << 8);
    u32 db = FCB::get_db(fs, FCBNum);
    u32 ndb = 0;
    if (size == 0)
        ndb = 1;
    else
        ndb = (size - 1) / fs->STORAGE_BLOCK_SIZE + 1;
    for (int i = 0; i < ndb; i++) {
        SUPERBLOCK::malloc_DB(fs, db + i);
    }
    return;
}

__device__ u32 FCB::search_by_filename(FileSystem* fs, char* filename) {
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        char currentfile[20] = "";
        FCB::get_filename(fs, i, currentfile);
        if (filename_equal(fs, filename, currentfile)) {
            return i;
        }
    }
    return -1;
}

__device__ u32 FCB::search_by_db(FileSystem* fs, u32 db) {
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        if (db == FCB::get_db(fs, i)) {
            return i;
        }
    }
    return -1;
}

__device__ u32 FCB::get_nFCB(FileSystem* fs) {
    for (int i = 0; i < fs->MAX_FILE_NUM; ++i) {
        if (fs->volume[i * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE] == '\0')
            return i;
    }
    return fs->MAX_FILE_NUM;
}

__device__ void FCB::move_FCB(FileSystem* fs, u32 desFCB, u32 tarFCB) {
    for (int i = 0; i < fs->FCB_SIZE; ++i) {
        u32 des = desFCB * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE + i;
        u32 tar = tarFCB * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE + i;
        fs->volume[des] = fs->volume[tar];
        fs->volume[tar] = '\0';
    }
    return;
}

__device__ void FCB::compact(FileSystem* fs, u32 FCBNum) {
    u32 db = FCB::get_db(fs, FCBNum);
    u32 nFCB = FCB::get_nFCB(fs);
    u32 nsize = FCB::get_size(fs, FCBNum);
    u32 ndb = 0;
    if (nsize == 0)
        ndb = 1;
    else
        ndb = (nsize - 1) / fs->STORAGE_BLOCK_SIZE + 1;
    for (int i = FCBNum + 1; i < nFCB; ++i) {
        u32 curdb = FCB::get_db(fs, i);
        u32 cursize = FCB::get_size(fs, u32(i));
        u32 curndb = (cursize - 1) / fs->STORAGE_BLOCK_SIZE + 1;
        for (int j = 0; j < curndb; ++j) {
            DB::move_DB(fs, curdb + j - ndb, curdb + j);
        }
        FCB::set_db(fs, i, curdb - ndb);
        FCB::move_FCB(fs, i - 1, i);
    }
    return;
}