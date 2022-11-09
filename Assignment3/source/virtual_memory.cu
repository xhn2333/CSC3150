#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "virtual_memory.h"

__device__ void init_invert_page_table(VirtualMemory* vm) {
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        vm->invert_page_table[i] = 0x80000000;  // invalid := MSB is 1     4KB
        vm->invert_page_table[i + vm->PAGE_ENTRIES] =
            i;  // Store page number    4KB
        vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] =
            0;  // Frequency clock, increase everytime, initial as 0    4KB
    }
}

__device__ void vm_init(VirtualMemory* vm,
                        uchar* buffer,
                        uchar* storage,
                        u32* invert_page_table,
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
    vm->pagefault_num_ptr = pagefault_num_ptr;

    // init constants
    vm->PAGESIZE = PAGESIZE;
    vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
    vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
    vm->STORAGE_SIZE = STORAGE_SIZE;
    vm->PAGE_ENTRIES = PAGE_ENTRIES;

    // before first vm_write or vm_read
    init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory* vm, u32 addr) {
    /* Complete vm_read function to read single element from data buffer */
    u32 pageNum = addr / vm->PAGESIZE;
    u32 offset = addr % vm->PAGESIZE;
    u32 address;

    // A clock, every read/write, the clock time will inrease 1
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if (!vm->invert_page_table[i] >> 31) {
            vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES]++;
        }
    }

    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        // "0" represent valid
        if ((vm->invert_page_table[i + vm->PAGE_ENTRIES] == pageNum) &&
            (!vm->invert_page_table[i])) {
            vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] =
                0;  // Clear the frequency clock
            address = i * vm->PAGESIZE + offset;
            return vm->buffer[address];
        }
    }

    // If there is no empty space, finad the least recently used block to swap
    // in
    u32 leastUsedPage = 0;
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if (vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] >
            vm->invert_page_table[leastUsedPage + 2 * vm->PAGE_ENTRIES]) {
            leastUsedPage = i;
        }
    }
    // Swap in or swap out
    for (int i = 0; i < vm->PAGESIZE; i++) {
        u32 storageSwapAddr =
            vm->invert_page_table[leastUsedPage + vm->PAGE_ENTRIES] *
                vm->PAGESIZE +
            i;
        u32 swapFrame = leastUsedPage * vm->PAGESIZE + i;
        u32 storageAddr = addr + i;

        vm->storage[storageSwapAddr] = vm->buffer[swapFrame];
        vm->buffer[swapFrame] = vm->storage[storageAddr];
    }
    // Refresh the page table
    vm->invert_page_table[leastUsedPage + vm->PAGE_ENTRIES] = pageNum;
    vm->invert_page_table[leastUsedPage + 2 * vm->PAGE_ENTRIES] =
        0;  // Clear the frequency clock
    address = leastUsedPage * vm->PAGESIZE + offset;

    return vm->buffer[address];
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value) {
    /* Complete vm_write function to write value into data buffer */
    u32 pageNum = addr / vm->PAGESIZE;
    u32 offset = addr % vm->PAGESIZE;
    u32 address;
    int is_exit;
    int signal = 1;

    // A clock, every read/write, the clock time will inrease 1
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if (!vm->invert_page_table[i] >> 31) {
            vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES]++;
        }
    }

    // Check wether the framw exita
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        // "0" represent valid
        if ((vm->invert_page_table[i + vm->PAGE_ENTRIES] == pageNum) &&
            (!vm->invert_page_table[i])) {
            vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] =
                0;  // Clear the frequency clock
            address = i * vm->PAGESIZE + offset;
            vm->buffer[address] = value;
            return;
        }
    }

    // Check whether the corresponding frame is empty
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        // "1" represent invalid;
        if (vm->invert_page_table[i] >> 31) {
            vm->invert_page_table[i] = 0;  // Refresh the page table valid bit
            *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;

            vm->invert_page_table[i + vm->PAGESIZE] = pageNum;
            vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] =
                0;  // Clear the frequency clock
            address = i * vm->PAGESIZE + offset;
            vm->buffer[address] = value;
            return;
        }
    }

    // If there is no empty space, finad the least recently used block to swap
    // in
    u32 leastUsedPage = 0;
    *(vm->pagefault_num_ptr) = *(vm->pagefault_num_ptr) + 1;
    for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
        if (vm->invert_page_table[i + 2 * vm->PAGE_ENTRIES] >
            vm->invert_page_table[leastUsedPage + 2 * vm->PAGE_ENTRIES]) {
            leastUsedPage = i;
        }
    }
    // Swap in or swap out
    for (int i = 0; i < vm->PAGESIZE; i++) {
        u32 storageSwapAddr =
            vm->invert_page_table[leastUsedPage + vm->PAGE_ENTRIES] *
                vm->PAGESIZE +
            i;
        u32 swapFrame = leastUsedPage * vm->PAGESIZE + i;
        u32 storageAddr = addr + i;

        vm->storage[storageAddr] = vm->buffer[swapFrame];
    }
    // Refresh the page table
    vm->invert_page_table[leastUsedPage + vm->PAGE_ENTRIES] = pageNum;
    vm->invert_page_table[leastUsedPage + 2 * vm->PAGE_ENTRIES] =
        0;  // Clear the frequency clock
    address = leastUsedPage * vm->PAGESIZE + offset;

    vm->buffer[address] = value;
}

__device__ void vm_snapshot(VirtualMemory* vm,
                            uchar* results,
                            int offset,
                            int input_size) {
    /* Complete snapshot function togther with vm_read to load elements from
     * data to result buffer */
    for (int i = 0; i < input_size; i++) {
        results[i] = vm_read(vm, i + offset);
    }
}
