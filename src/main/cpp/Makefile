CC=g++
CFLAGS=-g -Wall

INCDIR=/opt/include
INC_PARAMS=$(INCDIR:%=-I%)

LIBDIR=/opt/lib/xgboost /opt/lib/rabit
LIB_PARAMS=$(LIBDIR:%=-L%)
LIBS=-lxgboost

TARGET=xgboost_sharp

all: $(TARGET)

$(TARGET): $(TARGET).cpp
#	$(CC) $(CFLAGS) -o $(TARGET).o $(LIB) $(INC) $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET).o $(LIB_PARAMS) $(INC_PARAMS) $(TARGET).cpp $(LIBS)

clean:
	$(RM) $(TARGET)
