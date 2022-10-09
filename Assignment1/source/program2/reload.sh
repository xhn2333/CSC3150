#!/bin/bash

sudo rmmod program2 
make 
sudo insmod program2.ko 
sudo dmesg -c