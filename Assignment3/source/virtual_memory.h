#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <sys/types.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct Node {
    Node* prv;
    Node* nxt;
    int val;
};

struct VirtualMemory {
    uchar* buffer;
    uchar* storage;
    u32* invert_page_table;
    u32* invert_swap_table;
    int* pagefault_num_ptr;

    int PAGESIZE;
    int INVERT_PAGE_TABLE_SIZE;
    int PHYSICAL_MEM_SIZE;
    int STORAGE_SIZE;
    int PAGE_ENTRIES;
    int STORAGE_ENTRIES;
    Node* HEAD;
};

// TODO
__device__ void vm_init(VirtualMemory* vm,
                        uchar* buffer,
                        uchar* storage,
                        u32* invert_page_table,
                        u32* invert_swap_table,

                        int* pagefault_num_ptr,
                        int PAGESIZE,
                        int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE,
                        int STORAGE_SIZE,
                        int PAGE_ENTRIES);
__device__ uchar vm_read(VirtualMemory* vm, u32 addr);
__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory* vm,
                            uchar* results,
                            int offset,
                            int input_size);
__device__ u32 memSwap(VirtualMemory* vm, u32 pageNum);
__device__ u32 setLRU(VirtualMemory* vm);
__device__ void updateLRU(VirtualMemory* vm, u32 pageNum);
#endif
