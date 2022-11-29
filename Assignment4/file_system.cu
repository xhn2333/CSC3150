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
    SUPERBLOCK::update_storage_end_fp(fs, fs->FILE_BASE_ADDRESS);
}

__device__ u32 fs_open(FileSystem* fs, char* s, int op) {
    /* Implement open operation here */
    u32 fp = 0;
    int FCB_pos = FCB::search_file(fs, s);
    if (op == G_WRITE) {
        if (FCB_pos != -1)
            removeCompact(fs, FCB_pos);
        FCB_pos = FCB::create_FCB(fs, s, 0);
    }
    fp = FCB::get_fp(fs, FCB_pos);
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

    u32 FCB_pos = FCB::search_file_by_fp(fs, fp);
    if (fp + size > fs->STORAGE_SIZE + fs->FILE_BASE_ADDRESS) {
        removeCompact(fs, FCB_pos);
    } else {
        FCB::update_size(fs, FCB_pos, size);
        SUPERBLOCK::update_storage_end_fp(fs, FCB::get_fp(fs, FCB_pos) + size);
        // printf("fs_write ----- %d\n", fp);
        char s[20];
        FCB::get_filename(fs, FCB_pos, s);
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
        while (fs->volume[fs->SUPERBLOCK_SIZE + n * fs->FCB_SIZE] != '\0') {
            n++;
        }
        for (int i = n - 1; i >= 0; --i) {
            u32 FCB_pos = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
            char s[20] = "";
            FCB::get_filename(fs, FCB_pos, s);
            printf("%s\n", s);
        }
    } else if (op == LS_S) {
        printf("===sort by file size===\n");
        u32 entries[1024] = {};
        int n = 0;
        while (fs->volume[fs->SUPERBLOCK_SIZE + n * fs->FCB_SIZE] != '\0') {
            n++;
        }
        for (int i = 0; i < n; ++i) {
            u32 FCB_pos = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
            entries[i] = FCB_pos;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (FCB::get_size(fs, entries[i]) >=
                    FCB::get_size(fs, entries[j])) {
                    u32 t = entries[i];
                    entries[i] = entries[j];
                    entries[j] = t;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            u32 FCB_pos = entries[i];
            char s[20] = "";
            FCB::get_filename(fs, FCB_pos, s);
            printf("%s %d\n", s, FCB::get_size(fs, FCB_pos));
        }
    }
}

__device__ void fs_gsys(FileSystem* fs, int op, char* s) {
    /* Implement rm operation here */
    if (op == RM) {
        u32 FCB_pos = FCB::search_file(fs, s);
        removeCompact(fs, FCB_pos);
    }
    return;
}

/*
    FCB:
        0-19:   filename
        20-23:  size
        24-27:  fp
*/

__device__ u32 SUPERBLOCK::get_storage_end_fp(FileSystem* fs) {
    u32 s = fs->volume[3] + (fs->volume[2] << 8) + (fs->volume[1] << 16) +
            (fs->volume[0] << 24);
    return s;
}
__device__ void SUPERBLOCK::update_storage_end_fp(FileSystem* fs, u32 fp) {
    fs->volume[3] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[2] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[1] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[0] = fp % (1 << 8);
    return;
}

__device__ void removeCompact(FileSystem* fs, u32 FCB_pos) {
    u32 fp = FCB::get_fp(fs, FCB_pos);
    u32 size = FCB::get_size(fs, FCB_pos);
    int i = 0;
    int endfp = fs->FILE_BASE_ADDRESS;
    for (i = FCB_pos + fs->FCB_SIZE;
         i < fs->FILE_BASE_ADDRESS && fs->volume[i] != '\0';
         i += fs->FCB_SIZE) {
        u32 nowfp = FCB::get_fp(fs, u32(i));
        u32 nowsize = FCB::get_size(fs, u32(i));
        endfp = max(endfp, nowfp - size + nowsize);
        FCB::update_fp(fs, u32(i), nowfp - size);
        for (int j = 0; j < fs->FCB_SIZE; ++j)
            fs->volume[i + j - fs->FCB_SIZE] = fs->volume[i + j];
        for (int j = nowfp; j < nowfp + nowsize; ++j) {
            fs->volume[j - size] = fs->volume[j];
        }
    }
    for (int j = i - fs->FCB_SIZE; j < fs->FILE_BASE_ADDRESS; ++j) {
        fs->volume[j] = '\0';
    }
    SUPERBLOCK::update_storage_end_fp(fs, u32(endfp));

    return;
}

__device__ int FCB::search_file(FileSystem* fs, char* filename) {
    for (int i = fs->SUPERBLOCK_SIZE;
         i < fs->FILE_BASE_ADDRESS && fs->volume[i] != '\0';
         i += fs->FCB_SIZE) {
        char s[20];
        FCB::get_filename(fs, i, s);
        // printf("search_file ----- %s\n  current file name: %s\n", filename,
        // s);
        bool flag = true;
        for (int j = 0; j < fs->MAX_FILENAME_SIZE && filename[j] != '\0'; ++j) {
            if (filename[j] != s[j]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            // printf("  FCB_pos = %d\n", i);
            return i;
        }
    }
    return -1;
}

__device__ int FCB::search_file_by_fp(FileSystem* fs, u32 fp) {
    for (int i = fs->SUPERBLOCK_SIZE; i < fs->FILE_BASE_ADDRESS;
         i += fs->FCB_SIZE) {
        if (FCB::get_fp(fs, i) == fp) {
            return i;
        }
    }
    return -1;
}

__device__ u32 FCB::create_FCB(FileSystem* fs, char* filename, u32 size) {
    u32 FCB_pos = FCB::get_empty_FCB(fs);
    // printf("get_new_FCB ----- %d\n", FCB_pos);
    FCB::write_filename(fs, FCB_pos, filename);
    char s[20];
    FCB::get_filename(fs, FCB_pos, s);
    // printf("  file name: %s\n", s);
    FCB::update_size(fs, FCB_pos, size);
    return FCB_pos;
}

__device__ u32 FCB::get_empty_FCB(FileSystem* fs) {
    u32 FCB_pos = -1;
    for (int i = 0; i < fs->FCB_ENTRIES; ++i) {
        int idx = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
        if (fs->volume[idx] == '\0') {
            FCB_pos = idx;
            break;
        }
    }
    u32 fp = SUPERBLOCK::get_storage_end_fp(fs);
    FCB::update_fp(fs, FCB_pos, fp);
    return FCB_pos;
}

__device__ u32 FCB::get_size(FileSystem* fs, u32 FCB_pos) {
    u32 s = fs->volume[23 + FCB_pos] + (fs->volume[22 + FCB_pos] << 8) +
            (fs->volume[21 + FCB_pos] << 16) + (fs->volume[20 + FCB_pos] << 24);
    return s;
}

__device__ u32 FCB::get_fp(FileSystem* fs, u32 FCB_pos) {
    u32 s = fs->volume[27 + FCB_pos] + (fs->volume[26 + FCB_pos] << 8) +
            (fs->volume[25 + FCB_pos] << 16) + (fs->volume[24 + FCB_pos] << 24);
    return s;
}

__device__ void FCB::get_filename(FileSystem* fs, u32 FCB_pos, char* buf) {
    for (int i = 0;
         i < fs->MAX_FILENAME_SIZE && fs->volume[i + FCB_pos] != '\0'; ++i) {
        buf[i] = fs->volume[i + FCB_pos];
    }
    return;
}

__device__ u32 FCB::write_filename(FileSystem* fs,
                                   u32 FCB_pos,
                                   char* filename) {
    int i = 0;
    for (i = 0; i < fs->MAX_FILENAME_SIZE && filename[i] != '\0'; ++i) {
        fs->volume[i + FCB_pos] = filename[i];
    }
    for (int j = i; j < fs->MAX_FILENAME_SIZE; ++j) {
        fs->volume[j + FCB_pos] = '\0';
    }
    return 0;
}
__device__ void FCB::update_size(FileSystem* fs, u32 FCB_pos, u32 size) {
    fs->volume[23 + FCB_pos] = size % (1 << 8);
    size >>= 8;
    fs->volume[22 + FCB_pos] = size % (1 << 8);
    size >>= 8;
    fs->volume[21 + FCB_pos] = size % (1 << 8);
    size >>= 8;
    fs->volume[20 + FCB_pos] = size % (1 << 8);
    return;
}
__device__ void FCB::update_fp(FileSystem* fs, u32 FCB_pos, u32 fp) {
    fs->volume[27 + FCB_pos] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[26 + FCB_pos] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[25 + FCB_pos] = fp % (1 << 8);
    fp >>= 8;
    fs->volume[24 + FCB_pos] = fp % (1 << 8);
    return;
}
