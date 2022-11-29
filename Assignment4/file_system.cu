#include <bits/types/FILE.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "file_system.h"

__device__ __managed__ u32 gtime = 0;

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
                        int FILE_BASE_ADDRESS) {
    // init variables
    fs->volume = volume;

    // init constants
    fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
    fs->FCB_SIZE = FCB_SIZE;
    fs->FCB_ENTRIES = FCB_ENTRIES;
    fs->STORAGE_SIZE = VOLUME_SIZE;
    fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
    fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
    fs->MAX_FILE_NUM = MAX_FILE_NUM;
    fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
    fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op) {
    /* Implement open operation here */
    u32 fp = 0;
    u32 FCB_pos = FCB::search_by_filename(fs, s);
    if (op == G_WRITE) {
        if (FCB_pos != -1)
            FCB::compact(fs, FCB_pos);
        FCB_pos = FCB::new_FCB(fs, s);
    }
    u32 db = FCB::get_db(fs, FCB_pos);
    fp = DB::get_fp(fs, db);
    // printf("fs_open ----- %d\n", fp);
    return fp;
}

__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp) {
    /* Implement read operation here */
    for (int i = 0; i < size; ++i) {
        output[i] = fs->volume[fp + i];
    }
    return;
}

__device__ u32 fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp) {
    /* Implement write operation here */
    u32 db = DB::get_blockNum(fs, fp);
    u32 FCBNum = FCB::search_by_db(fs, db);
    if (fp + size > fs->STORAGE_SIZE + fs->FILE_BASE_ADDRESS) {
        FCB::compact(fs, FCBNum);
    } else {
        FCB::set_size(fs, FCBNum, size);
        // printf("fs_write ----- %d\n", fp);
        char s[20];
        FCB::get_filename(fs, FCBNum, s);
        // printf("  filename: %s\n  size: %d\n", s, FCB::get_size(fs,
        // FCB_pos)); printf("  end fp: %d\n",
        // SUPERBLOCK::get_storage_end_fp(fs)); printf("  input: ");
        for (int i = 0; i < size; ++i) {
            fs->volume[fp + i] = input[i];
            // printf("%c", input[i]);
        }
    }

    // printf("\n");
    return 0;
}
__device__ void fs_gsys(FileSystem* fs, int op) {
    /* Implement LS_D and LS_S operation here */
    if (op == LS_D) {
        printf("===sort by modified time===\n");
        int n = 0;
        n = FCB::get_nFCB(fs);
        for (int i = n - 1; i >= 0; --i) {
            char s[20] = "";
            FCB::get_filename(fs, i, s);
            printf("%s\n", s);
        }
    } else if (op == LS_S) {
        printf("===sort by file size===\n");
        u32 entries[1024] = {};
        int n = 0;
        n = FCB::get_nFCB(fs);
        for (int i = 0; i < n; ++i) {
            entries[i] = i;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (FCB::get_size(fs, entries[i]) >
                    FCB::get_size(fs, entries[j])) {
                    u32 t = entries[i];
                    entries[i] = entries[j];
                    entries[j] = t;
                } else if (FCB::get_size(fs, entries[i]) ==
                           FCB::get_size(fs, entries[j])) {
                    char s[20] = "", t[20] = "";
                    FCB::get_filename(fs, entries[i], s);
                    FCB::get_filename(fs, entries[j], t);
                    int cmp = filename_cmp(fs, s, t);
                    if (cmp == -1) {
                        u32 t = entries[i];
                        entries[i] = entries[j];
                        entries[j] = t;
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            char s[20] = "";
            FCB::get_filename(fs, entries[i], s);
            printf("%s %d\n", s, FCB::get_size(fs, entries[i]));
        }
    }
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) {
    /* Implement rm operation here */
    if (op == RM) {
        u32 FCBNum = FCB::search_by_filename(fs, s);
        FCB::compact(fs, FCBNum);
    }
    return;
}

/*
    FCB:
        0-19:   filename
        20-23:  size
        24-27:  fp
*/

__device__ bool filename_equal(FileSystem* fs, char* s, char* t) {
    int i = 0;
    while ((s[i] != '\0' || t[i] != '\0') && i < fs->MAX_FILENAME_SIZE) {
        if (s[i] != t[i])
            return false;
        i++;
    }
    return true;
}

__device__ int filename_cmp(FileSystem* fs, char* s, char* t) {
    int i = 0;
    while ((s[i] != '\0' && t[i] != '\0') && i < fs->MAX_FILENAME_SIZE) {
        if (s[i] > t[i])
            return 1;
        if (s[i] < t[i])
            return -1;
        i++;
    }
    return 0;
}