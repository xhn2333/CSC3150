#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "virtual_memory.h"

__device__ void init_invert_page_table(VirtualMemory* vm) {
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        vm->invert_page_table[i] = 0x80000000;  // invalid := MSB is 1
        vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    }
}

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
                        int PAGE_ENTRIES) {
    // init variables
    vm->buffer = buffer;
    vm->storage = storage;
    vm->invert_page_table = invert_page_table;
    vm->invert_swap_table = invert_swap_table;
    vm->pagefault_num_ptr = pagefault_num_ptr;

    // init constants
    vm->PAGESIZE = PAGESIZE;
    vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
    vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
    vm->STORAGE_SIZE = STORAGE_SIZE;
    vm->PAGE_ENTRIES = PAGE_ENTRIES;
    vm->STORAGE_ENTRIES = STORAGE_SIZE / PAGESIZE;
    // before first vm_write or vm_read
    init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory* vm, u32 addr) {
    /* Complate vm_read function to read single element from data buffer */
    uchar buf;

    int frameNum = -1;
    int firstEmptyPage = vm->PAGESIZE;

    u32 pageNum = addr / vm->PAGESIZE;
    u32 pageOffset = addr & (vm->PAGESIZE - 1);

    for (int i = 0; i < vm->PAGE_ENTRIES; ++i) {
        if (vm->invert_page_table[i] == pageNum) {
            buf = vm->buffer[i * vm->PAGESIZE + pageOffset];
            updateLRU(vm, i);
            // printf("%d\n", buf);
            return buf;
        }
    }  // TODO
    firstEmptyPage = memSwap(vm, pageNum);
    if (firstEmptyPage != 0xFFFFFFFF) {
        buf = vm->buffer[firstEmptyPage * vm->PAGESIZE + pageOffset];
        updateLRU(vm, firstEmptyPage);
        return buf;
    }
    return buf;
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value) {
    /* Complete vm_write function to write value into data buffer */
    int frameNum = -1;
    int firstEmptyPage = vm->PAGESIZE;

    u32 pageNum = addr / vm->PAGESIZE;
    u32 pageOffset = addr & ((vm->PAGESIZE) - 1);

    for (int i = 0; i < vm->PAGE_ENTRIES; ++i) {
        if (vm->invert_page_table[i] & 0x80000000) {
            firstEmptyPage = min(firstEmptyPage, i);
        } else if (vm->invert_page_table[i] == pageNum) {
            vm->buffer[i * vm->PAGESIZE + pageOffset] = value;
            updateLRU(vm, i);
            // printf("%d\n", value);
            return;
        }
    }
    firstEmptyPage = memSwap(vm, pageNum);
    if (firstEmptyPage != 0xFFFFFFFF) {
        vm->invert_page_table[firstEmptyPage] = pageNum;
        vm->buffer[firstEmptyPage * vm->PAGESIZE + pageOffset] = value;
        updateLRU(vm, firstEmptyPage);
    }
    return;
    // updateLRU(vm, firstEmptyPage);
}

__device__ u32 setLRU(VirtualMemory* vm) {
    int itr = 0;
    Node* now = vm->HEAD;
    while (now->nxt != NULL) {
        now = now->nxt;
        itr++;
    }
    return now->val;
}

__device__ void updateLRU(VirtualMemory* vm, u32 pageNum) {
    Node* now = vm->HEAD;
    if (now->val == pageNum) {
        return;
    }
    int itr = 0;
    bool hit = false;
    now = vm->HEAD;
    while (now->nxt != NULL) {
        if (pageNum == now->val) {
            Node* prev = now->prv;
            Node* nxt = now->nxt;
            now->prv = NULL;
            now->nxt = vm->HEAD;
            vm->HEAD = now;
            prev->nxt = nxt;
            nxt->prv = prev;
            hit = true;
            break;
        }
    }
    if (now->val == pageNum) {
        Node* prev = now->prv;
        prev->nxt = NULL;
        now->nxt = vm->HEAD;
        now->prv = NULL;
        hit = true;
    }
    if (!hit) {
        Node* tmp = new Node();
        tmp->val = pageNum;
        tmp->nxt = vm->HEAD;
        vm->HEAD = tmp;
    }
    return;
}

__device__ u32 memSwap(VirtualMemory* vm,
                       u32 pageNum) {  // return with a phyMemAddr
    (*(vm->pagefault_num_ptr))++;

    u32 victim = setLRU(vm);
    // Step 1: swap out victim
    bool hit = false;
    for (int i = 0; i < vm->STORAGE_ENTRIES; ++i) {
        if (vm->invert_swap_table[i] & 0x80000000) {
            vm->invert_swap_table[i] = vm->invert_page_table[victim];
            for (int j = 0; j < vm->PAGESIZE; ++j) {
                vm->storage[vm->PAGESIZE * i + j] =
                    vm->buffer[vm->PAGESIZE * victim + j];
            }

            // Step 2: change to invalid
            vm->invert_page_table[victim] &= 0x8FFFFFFF;
            hit = true;
            break;
        }
    }
    if (!hit) {
        printf("FATAL: Memory is full!\n");
        return 0xFFFFFFFF;
    }

    // Step 3: swap desired page
    for (int i = 0; i < vm->STORAGE_ENTRIES; ++i) {
        if (vm->invert_swap_table[i] == pageNum) {
            for (int j = 0; j < vm->PAGESIZE; ++j) {
                vm->buffer[vm->PAGESIZE * victim + j] =
                    vm->storage[vm->PAGESIZE * i + j];
                updateLRU(vm, victim);
            }
            break;
        }
    }

    // Step 4: reset page table for new page
    vm->invert_page_table[victim] = pageNum;
    return victim;
}

__device__ void vm_snapshot(VirtualMemory* vm,
                            uchar* results,
                            int offset,
                            int input_size) {
    /* Complete snapshot function togther with vm_read to load elements from
     * data to result buffer */
    for (int i = offset; i < input_size; i++) {
        results[i] = vm_read(vm, i);
    }
}
