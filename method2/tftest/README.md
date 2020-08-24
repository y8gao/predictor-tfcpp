# Build Application with Tensorflow C++ API

To build your C++ application with tensorflow C++ API, you may choose the second method, putting your codes within tensorflow code repo, i.e., `tensorflow/tensorflow/cc`.

Here is the steps to do:

- Create your project directory under `tensorflow/tensorflow/cc`.

  ```sh
  mkdir tensorflow/tensorflow/tftest
  ```

- Enter our project directory, write your codes.

- Prepare your bazel `BUILD` file.

- Build your project.

  ```sh
  bazel run -c opt //tensorflow/cc/tftest:tftest
  ```

## Reference

- https://tensorflow.juejin.im/api_guides/cc/guide.html

