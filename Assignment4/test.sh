rm test
nvcc --relocatable-device-code=true main.cu user_program.cu superblock.cu datablock.cu filecontrolblock.cu file_system.cu -o test
./test