
TF_HOME 	:= 3rd-party/tensorflow
ABSL_HOME   := 3rd-party/tensorflow/tensorflow/contrib/makefile/downloads/absl
PROTOBUF_HOME  := 3rd-party/protobuf/src


SRC_DIR := src
OBJ_DIR := obj

CUR_TARGET := $(notdir $(shell pwd))

OPT_FLAGS := -msse4.2 -mavx -mavx2 -mfma

LDFLAGS = -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w
LDLIBS = -L
INC_FILES=$(TF_HOME)/bazel-bin/

TF_INCLUDES 	:= 	-I$(TF_HOME) \
					-I$(TF_HOME)/bazel-bin \
					-I$(TF_HOME)/bazel-genfiles \
					-I$(TF_HOME)/bazel-tensorflow/external/eigen_archive \
					-I$(TF_HOME)/bazel-tensorflow/external/com_google_protobuf/src/ \
					-I$(TF_HOME)/bazel-tensorflow/external/com_google_absl/

TF_LIB_DIR    	:= 	$(TF_HOME)/bazel-bin/tensorflow
TF_LIBS     	:= 	-L$(TF_LIB_DIR) -ltensorflow_cc -ltensorflow_framework


#PROTOBUF_INCLUDES := -I$(PROTOBUF_HOME) -I$(PB_IFACE_HOME)
#PROTOBUF_LIB_DIR  := $(PROTOBUF_HOME)/.libs
#PROTOBUF_LIBS  	  := -L$(PROTOBUF_LIB_DIR) -lprotobuf



SRCFILES=$(wildcard $(SRCDIR)/*.cpp)

SOURCES 	:= $(wildcard $(SRC_DIR)/*.cpp)
OBJS 		:= $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))


CC		:= gcc
CXX 	:= g++
MKDIR	:= mkdir


all: $(CUR_TARGET)

$(CUR_TARGET): $(OBJS)
	@echo '==> Building target: $(CUR_TARGET)'
	@$(CXX) -o $@ $^ $(TF_LIBS)  $(PROTOBUF_LIBS)
	@echo 'Finished successfully building: $(CUR_TARGET)'
	@echo ' '

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@echo 'Building file: $<'
	@$(MKDIR) -p $(OBJ_DIR)
	@$(CXX) $(OPT_FLAGS) $(TF_INCLUDES) -I$(INC_DIR) -O0 -g -std=c++14 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:$(OBJ_DIR)/%.o=$(OBJ_DIR)/%.d)" -MT"$(@:$(OBJ_DIR)/%.o=$(OBJ_DIR)/%.d)" -o "$@" "$<"


clean:
	@$(RM) -rf ./$(OBJ_DIR)/*
	@$(RM) -rf ./$(CUR_TARGET)
	@echo 'Clean finished: $(CUR_TARGET)'
