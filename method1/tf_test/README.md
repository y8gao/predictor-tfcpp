# Build Application With Tensorflow C++ API

To build C++ application with tensorflow C++ API, please follow the following steps:

- Install required software packages.
- Clone tensorflow in directory `3rd-party`.
- Build tensorflow
  ```sh
  # enter tensorflow folder
  # then configure tensorflow
  ./configure

  # compile
  bazel build --config=opt //tensorflow:libtensorflow_cc.so

  ```
- Build application
  ```sh
  make clean
  make
  ```
