
To compile standalone resize_test2.cpp test program


$ g++ resize_test2.cpp -std=c++11
    `pkg-config --cflags --libs opencv4`
    -I/usr/include/rga -lrga -lstdc++ -o resize_test2



Then to run.

$ ./resize_test2 ../example/data/palace.jpg  /tmp/resize.jpg