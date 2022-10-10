#!/bin/bash

make
sudo insmod program2.ko 
sudo dmesg -c
sudo rmmod program2.ko